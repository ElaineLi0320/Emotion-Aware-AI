"""
    Training program for EmoNeXt model 
"""
import torch
from torch.optim import AdamW

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
            execution_name: ???
            lr: learning rate
            amp: automatic mixed precision
            ema_decay: exponential moving average decay
            ema_update_interval: ema update frequency
            gradient_accumu_steps: ???
            checkpoint_path: ???
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
        # [TODO] Write comments
        self.scaler = torch.amp.GradScaler(enabled=self.amp)
        self.scheduler = 

        
