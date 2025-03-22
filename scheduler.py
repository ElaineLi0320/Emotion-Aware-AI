"""
    A custom scheduler that implements warmup and restart
"""
import torch
import math
from torch.optim.lr_scheduler import LRScheduler

class CosineAnnealingWithWarmRestartsLR(LRScheduler):
    def __init__(self, optimizer:torch.optim.Optimizer, warmup_steps: int=128,
                 cycle_steps: int=512, min_lr:float=0.0, max_lr: float=1e-3):
        """
            warmup_steps: number of steps in warmup stage
            cycle_steps: total number of steps in a single cycle
            min_lr: minimum learning rate
            max_lr: maximum learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.min_lr = min_lr
        self.max_lr = max_lr
        
        # Keep track of current steps
        self.step_counter = 0
        super().__init__(optimizer)

    def step(self, epoch=None):
        self.step_counter += 1

        # Number of steps into current cycle
        current_cycle_steps = self.step_counter % self.cycle_steps
        
        # Linearly increase learning rate in warmup phase
        if current_cycle_steps < self.warmup_steps:
            current_lr = self.min_lr + (self.max_lr - self.min_lr) * current_cycle_steps / self.warmup_steps
        # Cosine decay phase
        else:
            current_lr = self.min_lr + (self.max_lr - self.min_lr) / 2 * (
                1 + math.cos(
                    math.pi * (current_cycle_steps - self.warmup_steps)
                    / (self.cycle_steps - self.warmup_steps)
                )
            )

        # Update each parameter group's learning rate
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = current_lr
