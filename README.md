# COOPMamba

COOPMamba is a vision model that combines **Mamba state space models** with **Context Optimization (CoOp)** for efficient and effective image classification.

## Overview

This project integrates two powerful approaches:

- **Mamba Architecture**: Efficient state space models (SSMs) that provide an alternative to Transformers, offering better scalability and computational efficiency for sequential data processing.
- **Context Optimization (CoOp)**: Learnable prompt-based classification that enables better few-shot and zero-shot learning capabilities.

## Features

- 🚀 Efficient vision backbone using Mamba state space models
- 🎯 CoOp-style learnable prompts for improved classification
- 📦 Modular architecture with easy configuration
- 🔧 Multiple model sizes (tiny, small, base, large)
- 📊 Training and evaluation scripts included
- ⚙️ Flexible configuration system

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA (optional, for GPU acceleration)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/npuNancy/COOPMamba.git
cd COOPMamba
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Note: `mamba-ssm` requires CUDA for installation. If you don't have CUDA, the code will fall back to a placeholder implementation.

## Project Structure

```
COOPMamba/
├── models/
│   ├── __init__.py
│   ├── mamba_block.py      # Mamba state space model blocks
│   ├── coop_module.py       # Context Optimization module
│   └── coopmamba.py         # Main COOPMamba model
├── configs/
│   └── default.yaml         # Default configuration
├── utils/
│   ├── __init__.py
│   ├── config.py            # Configuration utilities
│   ├── logger.py            # Logging utilities
│   └── metrics.py           # Evaluation metrics
├── scripts/
│   ├── train.py             # Training script
│   └── eval.py              # Evaluation/inference script
├── datasets/                # Dataset implementations (to be added)
├── requirements.txt
└── README.md
```

## Usage

### Training

To train a COOPMamba model:

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --data-path /path/to/dataset \
    --output-dir ./output
```

### Evaluation

To evaluate a trained model:

```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --data-path /path/to/dataset
```

### Inference

To run inference on a single image:

```bash
python scripts/eval.py \
    --config configs/default.yaml \
    --checkpoint /path/to/checkpoint.pth \
    --image /path/to/image.jpg
```

## Model Architecture

### Mamba Blocks

The vision encoder uses Mamba blocks for efficient feature extraction:
- State space model-based processing
- Bidirectional information flow
- Efficient linear complexity w.r.t. sequence length

### CoOp Module

The classification head uses learnable context vectors:
- `n_ctx` learnable context tokens
- Learnable class embeddings
- Cosine similarity-based classification

### Model Variants

| Model Size | Embed Dim | Depth | Parameters |
|-----------|-----------|-------|------------|
| Tiny      | 192       | 6     | ~10M       |
| Small     | 384       | 12    | ~30M       |
| Base      | 512       | 18    | ~70M       |
| Large     | 768       | 24    | ~150M      |

## Configuration

Edit `configs/default.yaml` to customize:

- Model architecture (size, dimensions, etc.)
- Training hyperparameters (learning rate, batch size, etc.)
- Data augmentation strategies
- Logging and checkpointing settings

## Example Code

```python
from models import create_coopmamba
import torch

# Create model
model = create_coopmamba(
    model_size='base',
    num_classes=1000,
    img_size=224,
    use_coop=True
)

# Forward pass
x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
logits = model(x)  # Output: (2, 1000)
```

## Performance

Performance metrics will be added after training on standard benchmarks.

## Citation

If you use COOPMamba in your research, please cite:

```bibtex
@software{coopmamba2025,
  title={COOPMamba: Combining Mamba State Space Models with Context Optimization},
  author={npuNancy},
  year={2025},
  url={https://github.com/npuNancy/COOPMamba}
}
```

## Acknowledgments

This project builds upon:
- [Mamba](https://github.com/state-spaces/mamba) - Efficient state space models
- [CoOp](https://arxiv.org/abs/2109.01134) - Context optimization for vision-language models
- [Vision Mamba](https://github.com/hustvl/Vim) - Vision applications of Mamba

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.