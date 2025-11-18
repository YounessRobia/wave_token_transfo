# Wavelet Neural Operator - Phase 2
## Wavelet Multi-Scale Attention for Vision Tasks

Complete implementation of Wavelet Transformers with scale-aware attention for image classification.

---

## 🎯 Overview

Phase 2 extends the Wavelet Neural Operator framework to vision tasks by introducing **Wavelet Multi-Scale Attention (W-MSA)**, a novel attention mechanism that operates directly in the wavelet domain.

### Key Features

- ✅ **2D Wavelet Tokenization**: Convert images to multi-scale wavelet tokens
- ✅ **Scale-Aware Positional Encoding**: Hierarchical position embeddings
- ✅ **PSD Kernel Design**: Positive semi-definite attention kernels
- ✅ **W-MSA Mechanism**: Multi-scale attention with locality constraints
- ✅ **Complete Transformer Architecture**: Full model with classification head
- ✅ **CIFAR-10 Experiments**: Training pipeline and evaluation
- ✅ **Analysis Tools**: Comparison with standard ViT

---

## 📦 What's Implemented

### Core Components (`wno/`)

1. **wavelets_2d.py** - 2D Wavelet Transform
   - Multi-level 2D DWT/IDWT
   - Perfect reconstruction
   - PyTorch gradient compatible

2. **tokenization.py** - Wavelet Tokenizer
   - Multi-scale token generation
   - Orientation-aware tokenization
   - Flexible pooling strategies

3. **positional_encoding.py** - Scale-Aware PE
   - Learnable scale embeddings
   - Orientation embeddings
   - Sinusoidal spatial encoding

4. **attention_kernel.py** - PSD Kernel
   - Scale locality kernel
   - Orientation similarity
   - Spatial attention with radius control

5. **wmsa.py** - W-MSA Module
   - Wavelet Multi-Scale Attention
   - Transformer blocks
   - Residual connections

6. **model.py** - Complete Architecture
   - Full Wavelet Transformer
   - Multiple model sizes
   - Classification head

7. **train_utils.py** - Training Utilities
   - Training/validation loops
   - Learning rate scheduling
   - Checkpoint management

### Experiments (`experiments/`)

1. **cifar10_classification.py** - Main Training Script
   - CIFAR-10 classification
   - Data augmentation
   - Full training pipeline

2. **compare_models.py** - Analysis Tools
   - Model comparison (W-MSA vs ViT)
   - Inference speed benchmarks
   - Feature visualization

---

## 🚀 Quick Start

### Installation

```bash
cd wno_phase2
pip install -r requirements.txt
```

### Test Installation

```python
python test_installation.py
```

### Train on CIFAR-10

**Quick test (tiny model, 10 epochs):**
```bash
cd experiments
python cifar10_classification.py --model-size tiny --epochs 10 --batch-size 128
```

**Full training (small model, 100 epochs):**
```bash
python cifar10_classification.py --model-size small --epochs 100 --batch-size 128 --lr 0.001
```

**With custom wavelet:**
```bash
python cifar10_classification.py --wavelet db4 --n-levels 3 --use-kernel
```

### Model Comparison

```bash
python compare_models.py
```

---

## 📊 Architecture Details

### Wavelet Transformer Pipeline

```
Input Image (32×32×3)
    ↓
2D Wavelet Transform (3 levels)
    ↓
Wavelet Coefficients: {cA, cH, cV, cD}
    ↓
Tokenization (each coeff → token)
    ↓
Scale-Aware Positional Encoding
    ↓
W-MSA Blocks (6 layers):
    ├─ Wavelet Multi-Scale Attention
    ├─ Layer Norm
    ├─ MLP
    └─ Residual Connection
    ↓
Global Pooling
    ↓
Classification Head (10 classes)
```

### Token Count Breakdown

For 32×32 image with 3 levels:

| Level | Scale | Tokens (A) | Tokens (H,V,D) | Total |
|-------|-------|------------|----------------|-------|
| 0     | 16×16 | 256        | 768            | 1024  |
| 1     | 8×8   | 64         | 192            | 256   |
| 2     | 4×4   | 1 (pooled) | 48             | 49    |

**Total Tokens: 1329** (vs 64 for ViT with 4×4 patches)

### Model Sizes

| Model | Embed Dim | Depth | Heads | Params | FLOPs |
|-------|-----------|-------|-------|--------|-------|
| Tiny  | 96        | 4     | 3     | ~1M    | ~0.5G |
| Small | 192       | 6     | 3     | ~4M    | ~2G   |
| Base  | 384       | 12    | 6     | ~16M   | ~8G   |

---

## 💡 Usage Examples

### Custom Model Configuration

```python
from wno.model import WaveletTransformer

model = WaveletTransformer(
    image_size=(32, 32),
    in_channels=3,
    num_classes=10,
    embed_dim=192,
    depth=6,
    num_heads=3,
    n_levels=3,
    wavelet='db4',
    use_kernel=True
)
```

### Feature Extraction

```python
# Extract multi-scale features
features, metadata = model.forward_features(images)

# features: [B, N, D] where N = number of tokens
# metadata: dict with scales, orientations, positions
```

### Training Loop

```python
from wno.train_utils import train_epoch, validate

for epoch in range(num_epochs):
    train_stats = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
    test_stats = validate(model, test_loader, criterion, device)
```

---

## 🧪 Ablation Studies

### Effect of PSD Kernel

```bash
# With kernel (W-MSA)
python cifar10_classification.py --use-kernel

# Without kernel (standard attention)
python cifar10_classification.py --no-use-kernel
```

### Effect of Wavelet Type

```bash
# Different wavelets
python cifar10_classification.py --wavelet db4
python cifar10_classification.py --wavelet sym4
python cifar10_classification.py --wavelet coif2
```

### Effect of Number of Levels

```bash
python cifar10_classification.py --n-levels 2
python cifar10_classification.py --n-levels 3
python cifar10_classification.py --n-levels 4
```

---

## 📚 Key Concepts

### 1. Wavelet Multi-Scale Attention

Unlike standard attention that treats all spatial locations equally, W-MSA:

- Restricts attention to nearby scales (scale bandwidth)
- Applies spatial locality within each scale
- Uses learnable orientation-wise interactions

### 2. PSD Kernel Design

The attention kernel is a product of three components:

```
K(τ, τ') = k_scale(j, j') × k_orient(ξ, ξ') × k_spatial(p, p')
```

- **k_scale**: RBF kernel on scale indices
- **k_orient**: Gram matrix on orientations
- **k_spatial**: Scale-normalized spatial kernel

### 3. Positional Encoding

Scale-aware encoding combines:

```
PE = PE_scale + PE_orient + PE_spatial
```

- Scale embeddings encode decomposition level
- Orientation embeddings distinguish H/V/D details
- Spatial encodings are normalized by scale factor

---

## 🔧 Advanced Configuration

### Command Line Arguments

```bash
python cifar10_classification.py \
  --model-size small \
  --n-levels 3 \
  --wavelet db4 \
  --epochs 100 \
  --batch-size 128 \
  --lr 0.001 \
  --weight-decay 0.05 \
  --warmup-epochs 5 \
  --save-dir ./checkpoints \
  --data-dir ./data
```

### Custom Scheduler

```python
from wno.train_utils import CosineAnnealingWarmupRestarts

scheduler = CosineAnnealingWarmupRestarts(
    optimizer,
    first_cycle_steps=100,
    max_lr=0.001,
    min_lr=0.00001,
    warmup_steps=5
)
```

---

## 📊 Performance Tips

### Training Speed

1. **Use smaller images**: 32×32 is optimal for CIFAR-10
2. **Reduce n_levels**: Fewer scales = fewer tokens
3. **Pool coarse approximation**: Reduces tokens significantly
4. **Mixed precision**: Use `torch.cuda.amp` for faster training

### Memory Usage

1. **Gradient checkpointing**: Trade compute for memory
2. **Smaller batch size**: If OOM, reduce to 64 or 32
3. **Model size**: Start with 'tiny' variant

### Accuracy Improvements

1. **Data augmentation**: RandAugment, Mixup, CutMix
2. **Longer training**: 200+ epochs with cosine schedule
3. **Larger model**: Use 'small' or 'base' variant
4. **Ensemble**: Average predictions from multiple models

---

## 📄 License

MIT License - Free for research and educational use.

---


