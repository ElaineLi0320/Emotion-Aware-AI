"""
    Prepare data and start model training
"""
import os
from tqdm import tqdm
import torch
from copy import deepcopy
from torchvision import transforms, datasets
from dotenv import load_dotenv
from models.Resnet import ResEmoteNet
import wandb
from datetime import datetime

# Load environment variables
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb_entity = os.getenv("WANDB_ENTITY")

# Define globla variables and hyperparameters
BASE_PATH = "data/"
BATCH_SIZE = int(input("Batch Size[default 16]: ").strip() or 16)
EPOCHS = int(input("Epochs[default 200]: ").strip() or 200)
DS_PATH = input("Dataset Path: ").strip()
LR = float(input("Optimizer Learning Rate[default 0.001]: ").strip() or 0.001)
MOMENTUM = float(input("Optimizer Momentum[default 0.9]: ").strip() or 0.9)
WEIGHT_DECAY = float(input("Optimizer Weight Decay[default 1e-4]: ").strip() or 1e-4)

# Initialize Weights&Biases logging
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
exec_name = f"ResEmoteNet_{current_time}"
wandb.init(project="ResEmoteNet", entity=wandb_entity, 
           name=exec_name, anonymous="allow",
           config={
                    "batch_size": BATCH_SIZE, 
                    "epochs": EPOCHS, 
                    "dataset_path": DS_PATH,
                    "optimizer_lr": LR,
                    "optimizer_momentum": MOMENTUM,
                    "optimizer_weight_decay": WEIGHT_DECAY
                  }
           )

FULL_PATH = os.path.join(BASE_PATH, DS_PATH)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("=========== Pre-training Info =============")
print(f"Using {device} device...\n")

# ============= Build a data transformation pipeline ============
train_transform = transforms.Compose([
    # transforms.Resize((64, 64)),
    # Faciliate the execution of SE Block
    transforms.Grayscale(num_output_channels=3),

     # 1. Intensity/Contrast adjustments (more appropriate for grayscale)
    transforms.RandomAdjustSharpness(sharpness_factor=1.5, p=0.3),
    
    # 2. Spatial transformations
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.1, 0.1),  # slight translation
        scale=(0.9, 1.1),  # slight scaling
    ),

    # 3. Noise and dropout (simulate different image qualities)
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.04)),  # reduced probability and scale
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.2)),
    
    # 4. Normalization
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

other_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# Load datasets using ImageFolder
train_ds = datasets.ImageFolder(os.path.join(FULL_PATH, "train"), train_transform)
val_ds = datasets.ImageFolder(os.path.join(FULL_PATH, "val"), other_transform)
test_ds = datasets.ImageFolder(os.path.join(FULL_PATH, "test"), other_transform)

# Create a dataloader for training, validation and test set
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = torch.utils.data.DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=True)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# Check their dimension
print(f"Training model with data from {FULL_PATH}.")
print(f"Train Samples: {len(train_ds)}")
print(f"Validation Samples: {len(val_ds)}")
print(f"Test Samples: {len(test_ds)}")

# Load our model
model = ResEmoteNet().to(device)

# Show total number of parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Model Total Params: {total_params:,}")

# Configure loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
# Set max gradient norm
max_grad_norm = 1.0
wandb.watch(model, log="all")

# ====================================== START TRAINING ==================================
# Keep track of loss and accuracy in different modes
training_losses = []
training_accuracies = []
validation_losses = []
validation_accuracies = []
test_losses = []
test_accuracies = []

# Record best validation accuracy
best_val_acc = 0

# Record total number of epochs in case of early termination
epoch_actual = 0

# Set maximum intervals between epochs for accuracy improvement before triggering early termination
max_interval = 15

# Cumulative interval
cumu_interval = 0

# Start training
for epoch in range(EPOCHS):
    print("\n=========== Training Info =============")
    # Activate training mode
    model.train()

    running_loss = 0
    # Number of correct predictions
    acc = 0
    # total number of samples
    total = 0

    for data in tqdm(train_dl, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        inputs, labels = data[0].to(device), data[1].to(device)

        # Reset gradients to zero
        optimizer.zero_grad()

        # Generate predictions
        output = model(inputs)

        # Compute loss
        loss = loss_fn(output, labels)

        # Initiate backpropagation
        loss.backward()

        # Add gradient clipping before optimizer step
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        # Update weights
        optimizer.step()

        # Update loss and accuracy for current epoch
        running_loss += loss.item()
        _, pred = torch.max(output.detach(), 1)
        acc += (pred == labels).sum().item()
        total += labels.size(0)

    # Record loss and accuracy in current epoch
    train_loss = running_loss / len(train_dl)
    train_acc = acc / total 
    training_losses.append(train_loss)
    training_accuracies.append(train_acc)

    # Activate evaluation mode for validation set
    model.eval()

    running_loss_val = 0
    acc_val = 0
    total_val = 0

    with torch.no_grad():
        for data in val_dl:
            inputs, labels = data[0].to(device), data[1].to(device)
            output = model(inputs)
            loss = loss_fn(output, labels)
            
            running_loss_val += loss.item()
            _, pred = torch.max(output.detach(), 1)
            acc_val += (pred == labels).sum().item()
            total_val += labels.size(0)

    val_loss = running_loss_val / len(val_dl)
    val_acc = acc_val / total_val
    validation_losses.append(val_loss)
    validation_accuracies.append(val_acc)

    # Activate evaluation for test set
    model.eval()

    running_loss_test = 0
    acc_test = 0
    total_test = 0

    with torch.no_grad():
        for data in test_dl:
            inputs, labels = data[0].to(device), data[1].to(device)
            output = model(inputs)
            loss = loss_fn(output, labels)

            running_loss_test += loss
            _, pred = torch.max(output.detach(), 1)
            acc_test += (pred == labels).sum().item()
            total_test += labels.size(0)
    
    test_loss = running_loss_test / len(test_dl)
    test_acc = acc_test / total_test
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

    # Log training and validation metrics to Weights&Biases
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_acc": train_acc,
        "val_loss": val_loss,
        "val_acc": val_acc, 
        "test_loss": test_loss,
        "test_acc": test_acc,
        "gradient_norm": torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    })

    print(f"Epoch {epoch}: train loss {train_loss:.4f}, train accuracy {train_acc:.4f};",
          f"test loss {test_loss:.4f}, test accuracy {test_acc:.4f}")
    epoch_actual += 1

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        cumu_interval = 0
        # Save a copy of state_dict of best performing model
        torch.save(deepcopy(model.state_dict()), "/result/best_model.pth")
    else:
        cumu_interval += 1
        print(f"No improvement for {cumu_interval} consecutive epochs.")

    if cumu_interval > max_interval:
        print(f"Stopping at {epoch_actual} epoch after no improvement for {max_interval} epochs.")
        break

# After the training loop ends, upload the best model to wandb:
print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
artifact = wandb.Artifact(name=f"model-{exec_name}", type="model")
artifact.add_file("/result/best_model.pth")
wandb.log_artifact(artifact)

wandb.finish()

        

