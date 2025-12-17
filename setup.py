"""
Setup script for COOPMamba
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

setup(
    name="coopmamba",
    version="0.1.0",
    author="npuNancy",
    description="COOPMamba: Combining Mamba State Space Models with Context Optimization",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/npuNancy/COOPMamba",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "tqdm>=4.65.0",
        "pyyaml>=6.0",
        "einops>=0.7.0",
        "timm>=0.9.0",
    ],
    extras_require={
        "mamba": ["mamba-ssm>=1.0.0"],
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "isort>=5.10.0",
        ],
    },
)
