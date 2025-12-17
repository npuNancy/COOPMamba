"""
Evaluation and inference script for COOPMamba
"""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image
import torchvision.transforms as transforms

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models import create_coopmamba
from utils import load_config, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate or run inference with COOPMamba')
    parser.add_argument('--config', type=str, default='configs/default.yaml',
                        help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to single image for inference')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to dataset for evaluation')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    return parser.parse_args()


def load_model(config, checkpoint_path, device):
    """
    Load model from checkpoint.
    """
    model = create_coopmamba(
        model_size=config['model']['size'],
        num_classes=config['model']['num_classes'],
        img_size=config['model']['img_size'],
        use_coop=config['model']['use_coop'],
        n_ctx=config['model']['n_ctx'],
        class_token_position=config['model']['class_token_position']
    )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model


def get_transforms(img_size=224):
    """
    Get image preprocessing transforms.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])


def inference_single_image(model, image_path, device, img_size=224):
    """
    Run inference on a single image.
    """
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(img_size)
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Run inference
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        top5_prob, top5_idx = torch.topk(probs, 5)
    
    print(f"\nTop 5 predictions for {image_path}:")
    for i in range(5):
        print(f"{i+1}. Class {top5_idx[0][i].item()}: {top5_prob[0][i].item()*100:.2f}%")
    
    return top5_idx[0][0].item(), top5_prob[0][0].item()


def evaluate_dataset(model, data_loader, device, logger):
    """
    Evaluate model on a dataset.
    """
    from utils import AverageMeter, accuracy
    
    model.eval()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            
            batch_size = images.size(0)
            top1.update(acc1.item(), batch_size)
            top5.update(acc5.item(), batch_size)
    
    logger.info(f'Evaluation Results: Top1={top1.avg:.2f}%, Top5={top5.avg:.2f}%')
    
    return top1.avg, top5.avg


def main():
    args = parse_args()
    
    # Setup logger
    logger = setup_logger()
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f'Configuration loaded from {args.config}')
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load model
    logger.info(f'Loading model from {args.checkpoint}')
    model = load_model(config, args.checkpoint, device)
    logger.info('Model loaded successfully')
    
    # Run inference or evaluation
    if args.image:
        # Single image inference
        inference_single_image(
            model, 
            args.image, 
            device, 
            config['model']['img_size']
        )
    elif args.data_path:
        # Dataset evaluation
        logger.warning('Dataset evaluation not fully implemented. Please add your dataset code.')
        # You would need to implement dataset loading here
    else:
        logger.error('Please specify either --image or --data-path')


if __name__ == '__main__':
    main()
