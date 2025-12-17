"""
Training script for COOPMamba
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import COOPMamba, create_coopmamba
from utils import load_config, setup_logger, AverageMeter, accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train COOPMamba')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--data-path', type=str, default='./data',
                        help='Path to dataset')
    parser.add_argument('--output-dir', type=str, default='./output',
                        help='Output directory for checkpoints')
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')
    parser.add_argument('--eval-only', action='store_true',
                        help='Only evaluate the model')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """
    Train for one epoch.
    """
    model.train()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (images, targets) in enumerate(pbar):
        images = images.to(device)
        targets = targets.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        # Update meters
        batch_size = images.size(0)
        losses.update(loss.item(), batch_size)
        top1.update(acc1.item(), batch_size)
        top5.update(acc5.item(), batch_size)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{losses.avg:.4f}',
            'top1': f'{top1.avg:.2f}',
            'top5': f'{top5.avg:.2f}'
        })
    
    logger.info(f'Train Epoch {epoch}: Loss={losses.avg:.4f}, Top1={top1.avg:.2f}%, Top5={top5.avg:.2f}%')
    
    return losses.avg, top1.avg, top5.avg


def validate(model, val_loader, criterion, device, logger):
    """
    Validate the model.
    """
    model.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Measure accuracy
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            # Update meters
            batch_size = images.size(0)
            losses.update(loss.item(), batch_size)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{losses.avg:.4f}',
                'top1': f'{top1.avg:.2f}',
                'top5': f'{top5.avg:.2f}'
            })
    
    logger.info(f'Validation: Loss={losses.avg:.4f}, Top1={top1.avg:.2f}%, Top5={top5.avg:.2f}%')
    
    return losses.avg, top1.avg, top5.avg


def main():
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logger
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(log_file=output_dir / 'train.log')
    
    logger.info(f'Configuration: {config}')
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Create model
    logger.info('Creating model...')
    model = create_coopmamba(
        model_size=config['model']['size'],
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        use_coop=config['model']['use_coop'],
        n_ctx=config['model']['n_ctx'],
        class_token_position=config['model']['class_token_position']
    )
    model = model.to(device)
    
    logger.info(f'Model created with {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters')
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Create loss function
    criterion = nn.CrossEntropyLoss()
    
    # Note: Dataset loading is omitted in this basic implementation
    # You would need to implement dataset loading based on your specific needs
    logger.warning('Dataset loading not implemented. Please add your dataset code.')
    
    logger.info('Training setup complete!')
    

if __name__ == '__main__':
    main()
