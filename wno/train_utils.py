"""
Training Utilities for Wavelet Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from typing import Dict, Optional, Tuple
import numpy as np


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output: torch.Tensor, target: torch.Tensor, topk=(1,)) -> list:
    """
    Computes the accuracy over the k top predictions
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def train_epoch(model: nn.Module,
                train_loader: DataLoader,
                criterion: nn.Module,
                optimizer: optim.Optimizer,
                device: torch.device,
                epoch: int,
                print_freq: int = 50,
                accumulation_steps: int = 1) -> Dict[str, float]:
    """
    Train for one epoch
    """
    model.train()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    batch_time = AverageMeter()
    
    end = time.time()
    
    for batch_idx, (images, targets) in enumerate(train_loader):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Scale loss for gradient accumulation
        loss = loss / accumulation_steps
        
        # Compute accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        # Backward pass
        loss.backward()
        
        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        
        # Update metrics (use unscaled loss for logging)
        losses.update(loss.item() * accumulation_steps, images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))
        batch_time.update(time.time() - end)
        end = time.time()
        
        # Print progress
        if batch_idx % print_freq == 0:
            print(f'Epoch: [{epoch}][{batch_idx}/{len(train_loader)}]\t'
                  f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                  f'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                  f'Acc@5 {top5.val:.2f} ({top5.avg:.2f})')
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg,
        'time': batch_time.sum
    }


def validate(model: nn.Module,
            val_loader: DataLoader,
            criterion: nn.Module,
            device: torch.device,
            print_freq: int = 50) -> Dict[str, float]:
    """
    Validate the model
    """
    model.eval()
    
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(val_loader):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Compute accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            # Update metrics
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            
            # Print progress
            if batch_idx % print_freq == 0:
                print(f'Test: [{batch_idx}/{len(val_loader)}]\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.2f} ({top1.avg:.2f})\t'
                      f'Acc@5 {top5.val:.2f} ({top5.avg:.2f})')
    
    print(f' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}')
    
    return {
        'loss': losses.avg,
        'acc1': top1.avg,
        'acc5': top5.avg
    }


class CosineAnnealingWarmupRestarts(optim.lr_scheduler._LRScheduler):
    """
    Cosine annealing with warmup and restarts
    """
    def __init__(self,
                 optimizer: optim.Optimizer,
                 first_cycle_steps: int,
                 cycle_mult: float = 1.0,
                 max_lr: float = 0.1,
                 min_lr: float = 0.001,
                 warmup_steps: int = 0,
                 gamma: float = 1.0,
                 last_epoch: int = -1):
        """
        Args:
            optimizer: Wrapped optimizer
            first_cycle_steps: Number of steps in first cycle
            cycle_mult: Cycle steps magnification
            max_lr: Maximum learning rate
            min_lr: Minimum learning rate
            warmup_steps: Linear warmup steps
            gamma: Decrease rate of max_lr after restart
            last_epoch: The index of last epoch
        """
        self.first_cycle_steps = first_cycle_steps
        self.cycle_mult = cycle_mult
        self.base_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        
        self.cur_cycle_steps = first_cycle_steps
        self.cycle = 0
        self.step_in_cycle = last_epoch
        
        super().__init__(optimizer, last_epoch)
        
        # Set initial learning rate
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def get_lr(self):
        if self.step_in_cycle == -1:
            return [self.min_lr for _ in self.base_lrs]
        elif self.step_in_cycle < self.warmup_steps:
            return [(self.max_lr - self.min_lr) * self.step_in_cycle / self.warmup_steps + self.min_lr 
                    for _ in self.base_lrs]
        else:
            return [self.min_lr + (self.max_lr - self.min_lr) * 
                    (1 + np.cos(np.pi * (self.step_in_cycle - self.warmup_steps) / 
                    (self.cur_cycle_steps - self.warmup_steps))) / 2
                    for _ in self.base_lrs]
    
    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.step_in_cycle = self.step_in_cycle + 1
            if self.step_in_cycle >= self.cur_cycle_steps:
                self.cycle += 1
                self.step_in_cycle = self.step_in_cycle - self.cur_cycle_steps
                self.cur_cycle_steps = int((self.cur_cycle_steps - self.warmup_steps) * 
                                           self.cycle_mult) + self.warmup_steps
        else:
            if epoch >= self.first_cycle_steps:
                if self.cycle_mult == 1.0:
                    self.step_in_cycle = epoch % self.first_cycle_steps
                    self.cycle = epoch // self.first_cycle_steps
                else:
                    n = int(np.log((epoch / self.first_cycle_steps * (self.cycle_mult - 1) + 1), 
                                   self.cycle_mult))
                    self.cycle = n
                    self.step_in_cycle = epoch - int(self.first_cycle_steps * 
                                                     (self.cycle_mult ** n - 1) / (self.cycle_mult - 1))
                    self.cur_cycle_steps = self.first_cycle_steps * self.cycle_mult ** (n)
            else:
                self.cur_cycle_steps = self.first_cycle_steps
                self.step_in_cycle = epoch
                
        self.max_lr = self.base_max_lr * (self.gamma ** self.cycle)
        self.last_epoch = epoch
        
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


def save_checkpoint(state: dict, 
                   is_best: bool,
                   filename: str = 'checkpoint.pth',
                   best_filename: str = 'model_best.pth'):
    """Save checkpoint to disk"""
    torch.save(state, filename)
    if is_best:
        torch.save(state, best_filename)


def load_checkpoint(model: nn.Module,
                    optimizer: Optional[optim.Optimizer] = None,
                    filename: str = 'checkpoint.pth',
                    device: torch.device = None) -> Tuple[int, float]:
    """
    Load checkpoint from disk
    
    Returns:
        start_epoch: Epoch to resume from
        best_acc: Best validation accuracy
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(filename, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])
    
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['best_acc']
    
    if optimizer is not None and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    print(f"Loaded checkpoint from epoch {start_epoch} with best acc {best_acc:.2f}%")
    
    return start_epoch, best_acc


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

