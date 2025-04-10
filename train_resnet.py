"""
    Prepare data and start model training
"""
import os
import torch
from tqdm import tqdm
import sys
import numpy as np
from copy import deepcopy
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from dotenv import load_dotenv
from models.Resnet import ResEmoteNet
import wandb
from datetime import datetime
import argparse
from sklearn.utils.class_weight import compute_class_weight

# Load environment variables
load_dotenv()
wandb_api_key = os.getenv("WANDB_API_KEY")
wandb_entity = os.getenv("WANDB_ENTITY")

# Set random seeds for reproducibility
seed = 2025
torch.manual_seed(seed)        # Controls PyTorch's CPU operations
torch.cuda.manual_seed(seed)   # Controls PyTorch's GPU operations
np.random.seed(seed)           # Controls NumPy's random operations
torch.backends.cudnn.deterministic = True  # Makes cuDNN operations deterministic
torch.backends.cudnn.benchmark = False     # Disables cuDNN's benchmarking

class Trainer:
    def __init__(self, model, train_dl, validation_dl, test_dl, classes, 
                 output_dir, max_epochs: int=200, early_stop: int=10,
                 execution_name=None, lr: float=0.001, momentum: float=0.9, 
                 weight_decay: float=1e-3):
        """
            model: ML model to train
            train_dl: training data loader
            validation_dl: validation data loader
            test_dl: test data loader
            classes: a list of class labels
            output_dir: program output directory
            max_epochs: maximum number of epochs
            early_stop: number of epochs without improvement before program stops prematurely
            execution_name: label for training run or model checkpoint
            lr: learning rate
            momentum: optimizer momentum
            weight_decay: optimizer weight decay
        """
        self.epochs = max_epochs
        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.test_dl = test_dl

        self.classes = classes
        self.num_classes = len(classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print(f"Device used: {self.device}")

        self.model = model.to(self.device)

        # Compute class weights
        class_labels = []
        for _, label in self.train_dl.dataset:
            class_labels.append(label)

        class_weights = compute_class_weight(class_weight="balanced", 
                                             classes=np.unique(class_labels), 
                                             y=class_labels)
        print(f"Class weights: {class_weights}")
        
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(self.device)

        # Configure loss function and optimizer
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)
        self.optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=lr, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
        
        self.early_stop_patience = early_stop
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.best_val_accuracy = 0
        self.execution_name = "model" if execution_name is None else execution_name

        # Connect Weights&Biases with model to monitor parameters and computational graph
        wandb.watch(model, log="all")

    def run(self):
        """
            Start the training process
        """
        # Record total number of epochs in case of early termination
        epoch_actual = 0

        # Cumulative interval for early stopping
        cumu_interval = 0

        # Start training
        for epoch in range(self.epochs):
            print(f"\n=========== Epoch {epoch+1}/{self.epochs} ============")
            
            # Train for one epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate for one epoch
            val_loss, val_acc = self.val_epoch()
            
            # Test for one epoch
            test_loss, test_acc = self.test_epoch()

            # Log training and validation metrics to Weights&Biases
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc, 
                "test_loss": test_loss,
                "test_acc": test_acc
            })

            print(f"Epoch {epoch}: train loss {train_loss:.4f}, train accuracy {train_acc:.4f};",
                  f"test loss {test_loss:.4f}, test accuracy {test_acc:.4f}")
            
            epoch_actual += 1

            # Early stopping logic
            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                cumu_interval = 0
                # Save a copy of state_dict of best performing model
                self.save()
            else:
                cumu_interval += 1
                print(f"No improvement for {cumu_interval} consecutive epochs.")

            if cumu_interval > self.early_stop_patience:
                print(f"Stopping at {epoch_actual} epoch after no improvement for {self.early_stop_patience} epochs.")
                break

        # After the training loop ends, upload the best model to wandb
        print(f"Training completed. Best validation accuracy: {self.best_val_accuracy:.4f}")
        artifact = wandb.Artifact(name=f"model-{self.execution_name}", type="model")
        artifact.add_file(os.path.join(self.output_dir, "best_model.pth"))
        wandb.log_artifact(artifact)

        wandb.finish()

    def train_epoch(self):
        """
            Train the model on batches of data
        """
        self.model.train()

        running_loss = 0
        # Number of correct predictions
        acc = 0
        # total number of samples
        total = 0

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.train_dl))
        for batch_idx, data in enumerate(self.train_dl):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)

            # Reset gradients to zero
            self.optimizer.zero_grad()

            # Generate predictions
            output = self.model(inputs)

            # Compute loss
            loss = self.loss_fn(output, labels)

            # Initiate backpropagation
            loss.backward()

            # Update weights
            self.optimizer.step()

            # Update loss and accuracy for current epoch
            running_loss += loss.item()
            _, pred = torch.max(output.detach(), 1)
            acc += (pred == labels).sum().item()
            total += labels.size(0)

            # Update progress bar
            pbar.set_postfix(
                {"Loss": running_loss / (batch_idx + 1), "ACC": (acc / total) * 100.0}
            )
            pbar.update()

        pbar.close()
        
        # Record loss and accuracy in current epoch
        train_loss = running_loss / len(self.train_dl)
        train_acc = acc / total
        
        return train_loss, train_acc

    def val_epoch(self):
        """
            Validate the model on the validation set
        """
        self.model.eval()

        running_loss_val = 0
        acc_val = 0
        total_val = 0

        with torch.no_grad():
            for data in self.validation_dl:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)
                
                running_loss_val += loss.item()
                _, pred = torch.max(output.detach(), 1)
                acc_val += (pred == labels).sum().item()
                total_val += labels.size(0)

        val_loss = running_loss_val / len(self.validation_dl)
        val_acc = acc_val / total_val
        
        return val_loss, val_acc

    def test_epoch(self):
        """
            Test the model on the test set
        """
        self.model.eval()

        running_loss_test = 0
        acc_test = 0
        total_test = 0

        with torch.no_grad():
            for data in self.test_dl:
                inputs, labels = data[0].to(self.device), data[1].to(self.device)
                output = self.model(inputs)
                loss = self.loss_fn(output, labels)

                running_loss_test += loss.item()
                _, pred = torch.max(output.detach(), 1)
                acc_test += (pred == labels).sum().item()
                total_test += labels.size(0)
        
        test_loss = running_loss_test / len(self.test_dl)
        test_acc = acc_test / total_test
        
        return test_loss, test_acc

    def save(self):
        """
            Save the model checkpoint
        """
        torch.save(
            deepcopy(self.model.state_dict()), 
            os.path.join(self.output_dir, "best_model.pth")
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ResEmoteNet model")

    parser.add_argument("--dataset_path", type=str, help="Path to the entire dataset")
    parser.add_argument("--output_dir", type=str, 
                        help="Path to save the model checkpoint", default="/result")
    parser.add_argument("--epochs", type=int, help="Maximum number of epochs(default=200)", default=200)
    parser.add_argument("--batch_size", type=int, help="Batch size for training(default=16)", default=16)
    parser.add_argument("--lr", type=float, help="Learning rate(default=0.001)", default=0.001)
    parser.add_argument("--momentum", type=float, help="Optimizer momentum(default=0.9)", default=0.9)
    parser.add_argument("--weight_decay", type=float, help="Optimizer weight decay(default=1e-3)", default=1e-3)
    parser.add_argument("--early_stop", type=int, help="Early stopping patience(default=10)", default=10)
    parser.add_argument("--num_workers", type=int, default=1,
                        help="The number of subprocesses to use for data loading(default=1)")

    opt = parser.parse_args()

    # If no dataset path is provided, ask for it
    if not opt.dataset_path:
        opt.dataset_path = input("Dataset Path: ").strip()

    # Define base path
    BASE_PATH = "data/"
    FULL_PATH = os.path.join(BASE_PATH, opt.dataset_path)

    # Initialize Weights&Biases logging
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exec_name = f"ResEmoteNet_{current_time}"
    
    # Always use wandb with the entity from .env file
    wandb.init(project="ResEmoteNet", 
               name=exec_name, 
               entity=wandb_entity,
               config={
                    "batch_size": opt.batch_size, 
                    "epochs": opt.epochs, 
                    "dataset_path": opt.dataset_path,
                    "optimizer_lr": opt.lr,
                    "optimizer_momentum": opt.momentum,
                    "optimizer_weight_decay": opt.weight_decay,
                    "early_stop": opt.early_stop
                }
    )

    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    print("=========== Pre-training Info =============")
    print(f"Using {device} device...\n")

    # ============= Build a data transformation pipeline ============
    train_transform = transforms.Compose([
        transforms.Resize((48, 48)),
        transforms.Grayscale(num_output_channels=3),
        
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=20),
        
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
    train_dl = DataLoader(
        train_ds, 
        batch_size=opt.batch_size, 
        shuffle=True,
        num_workers=opt.num_workers
    )
    val_dl = DataLoader(
        val_ds, 
        batch_size=opt.batch_size, 
        shuffle=False,
        num_workers=opt.num_workers
    )
    test_dl = DataLoader(
        test_ds, 
        batch_size=opt.batch_size, 
        shuffle=False,
        num_workers=opt.num_workers
    )

    # Check their dimension
    print(f"Training model with data from {FULL_PATH}.")
    print(f"Train Samples: {len(train_ds)}")
    print(f"Validation Samples: {len(val_ds)}")
    print(f"Test Samples: {len(test_ds)}")

    # Load our model
    model = ResEmoteNet()

    # Show total number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Total Params: {total_params:,}")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dl=train_dl,
        validation_dl=val_dl,
        test_dl=test_dl,
        classes=train_ds.classes,
        output_dir=opt.output_dir,
        max_epochs=opt.epochs,
        early_stop=opt.early_stop,
        execution_name=exec_name,
        lr=opt.lr,
        momentum=opt.momentum,
        weight_decay=opt.weight_decay
    )

    # Start training
    trainer.run()

        

