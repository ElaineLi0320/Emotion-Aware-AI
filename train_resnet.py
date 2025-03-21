"""
    Prepare data and start model training
"""
import os
from tqdm import tqdm
import torch
import pandas as pd
from copy import deepcopy
from torchvision import transforms, datasets
import time

from models.Resnet import ResEmoteNet

# Define globla variables used across this program
BASE_PATH = "data/"
BATCH_SIZE = 16
EPOCHS = 100
DS_PATH = input("Dataset Path: ").strip()
FULL_PATH = os.path.join(BASE_PATH, DS_PATH)

# Check for GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "mps")
print("=========== Pre-training Info =============")
print(f"Using {device} device...\n")

# ============= Build a data transformation pipeline ============
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    # Faciliate the execution of SE Block
    transforms.Grayscale(num_output_channels=3),
    # Introduces variability
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Load datasets using ImageFolder
train_ds = datasets.ImageFolder(os.path.join(FULL_PATH, "train"), transform)
val_ds = datasets.ImageFolder(os.path.join(FULL_PATH, "val"), transform)
test_ds = datasets.ImageFolder(os.path.join(FULL_PATH, "test"), transform)

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
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-4)

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

        # Generate perdictions
        output = model(inputs)

        # Compute loss
        loss = loss_fn(output, labels)

        # Initiate backpropagation
        loss.backward()

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

    print(f"Epoch {epoch}: train loss {train_loss:.4f}, train accuracy {train_acc:.4f};",
          f"test loss {test_loss:.4f}, test accuracy {test_acc:.4f}")
    epoch_actual += 1

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        cumu_interval = 0
        # Save a copy of state_dict of best performing  model instead of a reference
        torch.save(deepcopy(model.state_dict()), "/result/best_model.pth")
    else:
        cumu_interval += 1
        print(f"No improvement for {cumu_interval} consecutive epochs.")

    if cumu_interval > max_interval:
        print(f"Stopping at {epoch_actual} epoch after no improvement for {max_interval} epochs.")
        break

# Save all history info to a csv file
df = pd.DataFrame({
    "Epoch": range(1, epoch_actual+1),
    "Train Loss": [t.cpu().item() if torch.is_tensor(t) else t for t in training_losses],
    "Validation Loss": [t.cpu().item() if torch.is_tensor(t) else t for t in validation_losses],
    "Test Loss": [t.cpu().item() if torch.is_tensor(t) else t for t in test_losses],
    "Train Accuracy": [t.cpu().item() if torch.is_tensor(t) else t for t in training_accuracies],
    "Validation Accuracy": [t.cpu().item() if torch.is_tensor(t) else t for t in validation_accuracies],
    "Test Accuracy": [t.cpu().item() if torch.is_tensor(t) else t for t in test_accuracies]
})

time_stamp = time.strftime("%Y%m%d%H%M")
df.to_csv(f"/result/stats_{time_stamp}.csv", index=False)

        

