"""
    Training program for EmoNeXt model 
"""
import torch
import wandb 
from tqdm import tqdm
import sys
import numpy as np
from pathlib import Path
from scheduler import CosineAnnealingWithWarmRestartsLR
from torch.optim import AdamW
from ema_pytorch import EMA
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import torchvision
import random
import argparse
from datetime import datetime
from models.Emonext import get_model


seed = 2001
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Trainer:
    def __init__(self, model, train_dl, validation_dl, test_dl, classes, 
                 output_dir, max_epochs: int=10000, early_stop: int=12,
                 execution_name=None, lr: float=1e-4, amp: bool=False, ema_decay: float=0.99,
                 ema_update_interval: int=16, gradient_accumu_steps: int=1, 
                 checkpoint_path: str=None):
        """
            model: ML model to train
            train_dl: training data loader
            validation_dl: validation data loader
            test_dl: test data loader
            classes: a list of class lables
            output_dir: program output directory
            max_epochs: maximum number of epochs
            early_stop: number of epochs without improvement before program stops prematurely
            execution_name: label for training run or model checkpoint
            lr: learning rate
            amp: automatic mixed precision
            ema_decay: exponential moving average decay
            ema_update_interval: ema update frequency
            gradient_accumu_steps: number of steps before gradients update is triggered
            checkpoint_path: file path to saved models
        """
        self.epochs = max_epochs
        self.train_dl = train_dl
        self.validation_dl = validation_dl
        self.test_dl = test_dl

        self.classes = classes
        self.num_classes = len(classes)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        print(f"Device used: {self.device}")

        self.amp = amp
        self.gradient_accumu_steps = gradient_accumu_steps

        self.model = model.to(self.device)

        self.optimizer = AdamW(model.parameters(), lr=lr)
        # Scaling factor that prevents gradient from underflowing
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        # Learning rate scheduler
        self.scheduler = CosineAnnealingWithWarmRestartsLR(
            self.optimizer, warmup_steps=128, cycle_steps=1024
        )

        self.ema = EMA(model, 
                       beta=ema_decay, # ema factor
                       update_every=ema_update_interval).to(self.device)
        
        self.early_step_patience = early_stop
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        self.best_val_accuracy = 0
        self.execution_name = "model" if execution_name is None else execution_name

        # Load saved model checkpoint
        if checkpoint_path:
            self.load(checkpoint_path)

        # Connect Weights&Biases with model to monitor parameters and computational graph
        wandb.watch(model, log="all")

    def run(self):
        """
            Start the training process
        """
        # Counter for epochs with no validation loss improvement
        counter = 0

        images, _ = next(iter(self.train_dl))
        # Transform tensor image to PIL image
        images = [transforms.ToPILImage()(image) for image in images]
        # Log images on Weights&Biases
        wandb.log({"Images": [wandb.Image(image) for image in images]})

        for epoch in range(self.epochs):
            print(f"Epochs: {epoch+1}/{self.epochs}")

            self.visualize_stn()
            train_loss, train_accuracy = self.train_epoch()
            val_loss, val_accuracy = self.val_epoch()

            wandb.log(
                {
                    "Train Loss": train_loss,
                    "Val Loss": val_loss,
                    "Train Accuracy": train_accuracy,
                    "Val Accuracy": val_accuracy,
                    "Epoch": epoch + 1
                }
            )

            # Early stopping if no improvement after certain epochs
            if val_accuracy > self.best_val_accuracy:
                self.save()
                counter = 0
                self.best_val_accuracy = val_accuracy
            else:
                counter += 1
                print(f"Validation loss didn't improve for {counter} epochs.")
                if counter >= self.early_step_patience:
                    print(f"Validation didn't improve for {self.early_step_patience} epochs. Training stopped.")
                    break

        self.test_model()
        wandb.finish()

    def train_epoch(self):
        """
            Train the model on batches of data
        """
        self.model.train()

        avg_accuracy = []
        avg_loss = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.train_dl))
        # Training on a batch of data
        for batch_idx, data in enumerate(self.train_dl):
            inputs, labels = data

            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Enable automatic precision adjustment
            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)
            
            # Scales loss by a constant factor and compute gradients
            self.scaler.scale(loss).backward()
            # Accumulate gradients for a fixed number of steps
            if (batch_idx + 1) % self.gradient_accumu_steps == 0:
                # Clip the scaled gradient
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                # Unscale gradient to prevent distortion
                self.scaler.step(self.optimizer)
                # Clear computed gradients and set them to None
                self.optimizer.zero_grad(set_to_none=True)
                # Adjust scaling factor accordingly
                self.scaler.update()
                self.ema.update()
                # Update learning rate according to preset schedule
                self.scheduler.step()

            # Compute average accuracy of current batch
            batch_accuracy = (predictions == labels).sum().item() / labels.size(0)

            # Append accuracy and loss of current batch
            avg_accuracy.append(batch_accuracy)
            avg_loss.append(loss.item())

            # Update progress bar
            pbar.set_postfix(
                {"Loss": np.mean(avg_loss), "ACC": np.mean(avg_accuracy) * 100.0}
            )
            pbar.update()

        pbar.close()
        return np.mean(avg_loss), np.mean(avg_accuracy)

    def val_epoch(self):
        """
            Validate the model on the validation set
        """
        self.model.eval()

        avg_loss = []
        predicted_lables = []
        true_labels = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.validation_dl))

        for batch_idx, data in enumerate(self.validation_dl):
            inputs, labels = data
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            with torch.autocast(self.device.type, enabled=self.amp):
                predictions, _, loss = self.model(inputs, labels)

            avg_loss.append(loss.item())
            predicted_lables.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update()
        
        pbar.close()

        # Compute accuracy of the entire validation set
        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels)).float().mean().item()
        )
        
        wandb.log(
            {
                "Confusion Matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=predicted_labels,
                    class_names=self.classes
                )
            }
        )

        print(
            f"Validation Loss: {np.mean(avg_loss) * 1.0:.4f}, "
            f"Validation Accuracy: {accuracy * 100.0:.4f}%"
        )

        return np.mean(avg_loss), accuracy * 100.0

    def test_model(self):
        """
            Test the model on the test set
        """
        self.model.eval()

        predicted_labels = []
        true_labels = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.test_dl))
        for batch_idx, (inputs, labels) in enumerate(self.test_dl):
            bs, ncrops, c, h, w = inputs.shape
            inputs = inputs.view(-1, c, h, w).to(self.device)
            labels = labels.to(self.device)

            # Enable automatic precision adjustment
            with torch.autocast(self.device.type, enabled=self.amp):
                # Use EMA-smoothed model for testing
                _, logits = self.ema(inputs)
            outputs_avg = logits.view(bs, ncrops, -1).mean(dim=1)
            predictions = torch.argmax(outputs_avg, dim=1)

            predicted_labels.extend(predictions.tolist())
            true_labels.extend(labels.tolist())

            pbar.update()
        
        pbar.close()

        # Compute accuracy of the entire test set
        accuracy = (
            torch.eq(torch.tensor(predicted_labels), torch.tensor(true_labels)).float().mean().item()
        )

        print(f"Test Accuracy: {accuracy * 100.0:.4f}%")

        wandb.log(
            {
                "Confusion Matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=true_labels,
                    preds=predicted_labels,
                    class_names=self.classes
                )
            }
        )

        
    def visualize_stn(self):
        """
            Visually compare original images and transformed images after STN
        """
        self.model.eval()
        
        # Get a batch of images for visualization
        batch = torch.utils.data.Subset(val_dataset, range(32))

        # Stack images together
        batch = torch.stack([batch[i][0] for i in range(len(batch))]).to(self.device)
        with torch.autocast(self.device.type, enabled=self.amp):
            stn_batch = self.model.stn(batch)

        # Create a transformation pipeline
        to_pil = transforms.ToPILImage()
        
        # Original images 
        grid = to_pil(torchvision.utils.make_grid(batch, nrow=16, padding=4))
        # Transformed images
        stn_batch = to_pil(torchvision.utils.make_grid(stn_batch, nrow=16, padding=4))

        # Log images on Weights&Biases
        wandb.log({"Batch": wandb.Image(grid), "Transformed Images": wandb.Image(stn_batch)})

    def save(self):
        """
            Save the model checkpoint
        """
        data = {
            "model": self.model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "ema": self.ema.state_dict(),
            "scaler": self.scaler.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "best_acc": self.best_val_accuracy,
        }

        torch.save(data, str(self.output_dir / f"{self.execution_name}.pt"))

    def load(self, path):
        """
            Load the model checkpoint

            path: path to the model checkpoint
        """
        # Load model checkpoint to the device
        data = torch.load(path, map_location=self.device)

        self.model.load_state_dict(data["model"])
        self.optimizer.load_state_dict(data["opt"])
        self.ema.load_state_dict(data["ema"])
        self.scaler.load_state_dict(data["scaler"])
        self.scheduler.load_state_dict(data["scheduler"])
        self.best_val_accuracy = data["best_acc"]

def plot_images():
    # Create a grid of images for visualization from the training dataset
    num_rows = 4
    num_cols = 8
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(10, 5))

    # Plot the images
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            image, _ = train_dataset[index]
            # Convert tensor to PIL image format
            axes[i, j].imshow(image.permute(1, 2, 0))
            axes[i, j].axis("off")

    plt.tight_layout()
    plt.savefig(f"images.png")
    plt.close()

# Custom transform to repeat the grayscale image channels to 3
class RepeatChannels:
    def __call__(self, x):
        return x.repeat(3, 1, 1)

# Custom transform to stack the tensor crops
class StackTensorCrops:
    def __init__(self):
        self.to_tensor = transforms.ToTensor()
    
    def __call__(self, crops):
        return torch.stack([self.to_tensor(crop) for crop in crops])

# Custom transform to repeat the channels of the tensor crops
class RepeatChannelsCrops:
    def __call__(self, crops):
        return torch.stack([crop.repeat(3, 1, 1) for crop in crops])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EmoNeXt model")

    parser.add_argument("--dataset_path", type=str, help="Path to the training dataset")
    parser.add_argument("--output_dir", type=str, 
                        help="Path to save the model checkpoint", default="out")

    parser.add_argument("--epochs", type=int, help="Maximum number of epochs")
    parser.add_argument("--batch_size", type=int, help="Batch size for training", default=32)

    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument(
        "--amp",
        action="store_true", # Set to true if this argument is provided
        default=False,
        help="Enable mixed precision training",
    )

    # Whether to use 22k pre-trained weights
    parser.add_argument("--in_22k", action="store_true", default=False)
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of steps to accumulate gradients before updating the model weights",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="The number of subprocesses to use for data loading."
        "0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to the checkpoint file for resuming training or performing inference",
    )
    parser.add_argument(
        "--model_size",
        choices=["tiny", "small", "base", "large", "xlarge"],
        default="tiny",
        help="Choose the size of the model: tiny, small, base, large, or xlarge",
    )

    opt = parser.parse_args()
    print(opt)

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exec_name = f"EmoNeXt_{opt.model_size}_{current_time}"

    wandb.init(project="EmoNeXt", name=exec_name, anonymous="must")

    # Define transformations for training, validation, and testing
    train_transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomRotation(degrees=20),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            RepeatChannels(),
        ]
    )

    val_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            RepeatChannels(),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Grayscale(),
            transforms.Resize(236),
            transforms.TenCrop(224),
            StackTensorCrops(),
            RepeatChannelsCrops(),
        ]
    )

    train_dataset = datasets.ImageFolder(opt.dataset_path + "/train", train_transform)
    val_dataset = datasets.ImageFolder(opt.dataset_path + "/val", val_transform)
    test_dataset = datasets.ImageFolder(opt.dataset_path + "/test", test_transform)

    print("Using %d images for training." % len(train_dataset))
    print("Using %d images for evaluation." % len(val_dataset))
    print("Using %d images for testing." % len(test_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.num_workers,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    net = get_model(len(train_dataset.classes), opt.model_size, in_22k=opt.in_22k)

    trainer = Trainer(
        model=net,
        train_dl=train_loader,
        validation_dl=val_loader,
        test_dl=test_loader,
        classes=train_dataset.classes,
        execution_name=exec_name,
        lr=opt.lr,
        output_dir=opt.output_dir,
        max_epochs=opt.epochs,
        amp=opt.amp,
        checkpoint_path=opt.checkpoint,
    )

    trainer.run()

