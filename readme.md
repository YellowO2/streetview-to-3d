# google-map-to-3d

## Requirements

- NVIDIA GPU with driver version ≥570 (to support CUDA 13.0)
- Python 3.12

## Setup

**1. Install PyTorch for CUDA 13.0 first**

```bash
pip install torch==2.11.0 torchvision==0.26.0 triton==3.6.0 xformers==0.0.35 --index-url https://download.pytorch.org/whl/cu130
```

If your driver doesn't support CUDA 13.0, check [pytorch.org](https://pytorch.org/get-started/locally/) to find the right version for your system.

**2. Install the rest of the dependencies**

```bash
pip install -r requirements.txt
```

**3. Configure model paths**

Edit `config.yaml` and set the absolute paths to the model files on your machine:

```yaml
models:
  sharp: /absolute/path/to/models/sharp_2572gikvuh.pt
  da3: /absolute/path/to/models--depth-anything--DA3NESTED-GIANT-LARGE-1.1/snapshots/b2359bdf726fb44ef62acca04d629dcf158053e7
```

**4. Run**

```bash
uvicorn main:app --reload
```
