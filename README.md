# COOPMamba: Efficient Vehicle-to-Vehicle Cooperative Perception based on 3D Point Clouds

This repository is the official implementation of the paper **"CoopMamba"**.

CoopMamba is a novel cooperative perception framework for autonomous driving that leverages **State Space Models (Mamba)** to efficiently fuse features from multiple connected autonomous vehicles (CAVs).

## 🔥 Features
- **Mamba-based Fusion**: Introduces `COOPMamba`, a fusion module utilizing `SingleMambaBlock` and `CrossMambaBlock` for effective spatial and cross-agent feature interaction.
- **Efficient**: Leverages the linear complexity of Mamba for efficient processing of large-scale cooperative perception data.
- **SOTA Performance**: Achieves state-of-the-art performance on the OPV2V dataset.

## 🛠️ Installation

Please follow the installation steps below to set up the environment.

### 1. Create Conda Environment
```bash
conda create -n coopmamba python=3.8
conda activate coopmamba
```

### 2. Install PyTorch
Install PyTorch with CUDA support (tested with CUDA 11.8).
```bash
pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu118
```

### 3. Install Mamba-SSM
CoopMamba requires `mamba_ssm` and `causal_conv1d`.
```bash
pip install causal-conv1d>=1.2.0
pip install mamba-ssm
```

### 4. Install Other Dependencies
```bash
pip install -r COOPMamba_requirements.txt
python setup.py develop
```

### 5. Install Spconv
```bash
pip install spconv-cu118
```

## 📊 Data Preparation

Please refer to the [OpenCOOD Data Preparation](https://opencood.readthedocs.io/en/latest/md_files/data_intro.html) guide to download and prepare the **OPV2V** dataset.

Organize the data as follows:
```
opv2v/
  train/
  validate/
  test/
```

Update the `root_dir` and `validate_dir` in `opencood/hypes_yaml/point_pillar_COOPMamba_opv2v.yaml` to point to your dataset path.

## 🚀 Training

To train the CoopMamba model on the OPV2V dataset, run the following command:

```bash
python opencood/tools/train.py --hypes_yaml opencood/hypes_yaml/point_pillar_COOPMamba_opv2v.yaml
```

Arguments:
- `--hypes_yaml`: Path to the configuration file.
- `--model_dir` (Optional): Directory to save the model checkpoints. Default is `opv2v_data_dumping/`.

## 🧪 Evaluation

To evaluate the trained model, use the following command:

```bash
python opencood/tools/inference.py --hypes_yaml opencood/hypes_yaml/point_pillar_COOPMamba_opv2v.yaml --model_dir opv2v_data_dumping/train/point_pillar_COOPMamba --fusion_method intermediate
```

## 📜 Acknowledgement

This project is based on [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD). We thank the authors for their excellent work.

