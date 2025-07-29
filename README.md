# PhysMorph: A Biomechanical and Image-Guided Deep Learning Framework for Real-Time Multi-Modal Liver Image Registration

## Overview
This repository contains the implementation of a 3D U-Net architecture with self-attention mechanism for deformable image registration between Cone-Beam CT (CBCT) and Magnetic Resonance (MR) images. The model predicts dense displacement vector fields (DVF) to align multi-modal medical images.

### Key Features
- **3D U-Net Architecture**: Encoder-decoder structure with skip connections
- **Self-Attention Module**: Attention mechanism in the bottleneck layer for better feature representation
- **Spatial Transformation Layer**: Applies predicted DVF to warp images
- **Multi-component Loss**: Combines DVF regression loss and image similarity loss
- **Liver Region Masking**: Automatic extraction and refinement of liver regions

## Architecture

### Model Components
1. **Encoder**: 4-level downsampling with double convolution blocks
2. **Bottleneck**: Enhanced with 3D self-attention mechanism
3. **Decoder**: 4-level upsampling with skip connections
4. **Output**: 3-channel DVF prediction (x, y, z displacements)

### Loss Function
- DVF Mean Squared Error (MSE)
- Image similarity loss (optional)
- Weighted combination: `L_total = α·L_DVF + β·L_similarity`

## Installation

### Requirements
- Python ≥ 3.8
- CUDA-capable GPU (recommended)

### Setup
```bash
# Clone repository
git clone https://github.com/DongyangGu0/PhysMorph.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Data Format

The model expects medical imaging data in MATLAB `.mat` format:

```
dataset/
├── train/
│   ├── CBCT/CT_*.mat    # CBCT volumes
│   ├── MR/MR_*.mat      # MR volumes  
│   └── DVF/dvf_*.mat    # Ground truth DVFs
└── test/
    ├── CBCT/CT_*.mat
    └── MR/MR_*.mat
```

### Data Structure
- **CBCT/MR volumes**: 3D arrays with shape `(H, W, D)`
- **DVF**: 4D arrays with shape `(H, W, D, 3)` representing (x, y, z) displacements

## Usage

### Configuration
Edit `config.py` to set your data paths and training parameters:

```python
# Data paths
cbct_dir = "/path/to/CBCT"
mr_dir = "/path/to/MR"
dvf_dir = "/path/to/DVF"

# Training parameters
batch_size = 1
learning_rate = 1e-4
total_epochs = 5000
```

### Training
```bash
python train.py
```

The training script will:
- Load and preprocess data
- Train the model with periodic validation
- Save checkpoints every N epochs
- Generate visualization results

### Inference
```bash
python test.py
```

This will:
- Load a trained model checkpoint
- Process test data
- Output predicted DVFs and warped images

## Project Structure
```
.
├── config.py              # Configuration parameters
├── train.py               # Training script
├── test.py                # Inference script
├── unet3d_model.py        # Model architecture
├── self_attention.py      # Alternative model implementation
├── inspect_mat.py         # Utility for .mat file inspection
├── requirements.txt       # Python dependencies
├── README.md              # This file
└── .gitignore            # Git ignore rules
```

## Model Performance

### Training Details
- Input: Concatenated CBCT and MR volumes (2 channels)
- Output: 3D displacement vector field
- Optimizer: Adam
- Learning rate: 1e-4
- Training epochs: 5000

### Evaluation Metrics
- DVF prediction accuracy (MSE)
- Image alignment quality
- Computational efficiency

## Citation

If you use this code in your research, please cite:

```bibtex
@article{
  title={PhysMorph: A Biomechanical and Image-Guided Deep Learning Framework for Real-Time Multi-Modal Liver Image Registration},
  author={Zeyu Zhang, Dongyang Guo, et al.},
  journal={Journal Name},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- PyTorch development team
- Medical imaging community for valuable insights

## Contact

For questions or collaborations, please contact: [zeyu.zhang@duke.edu; dongyang.guo@duke.edu]

---

**Note**: This repository contains only the source code. Medical imaging data is not included due to privacy and ethical considerations. Users must provide their own data following the specified format. 
