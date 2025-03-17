"""
    Training program for EmoNeXt model 
"""
import torch
import wandb 
import tqdm
import sys
import numpy as np
from pathlib import Path
from scheduler import CosineAnnealingWithWarmRestartsLR
from torch.optim import AdamW
from ema_pytorch import EMA
from torchvision import datasets, transforms

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
            Start model training
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
        self.model.train()

        avg_accuracy = []
        avg_loss = []

        pbar = tqdm(unit="batch", file=sys.stdout, total=len(self.training_dataloader))
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
