## Installation
See [installation instructions](https://xmodaler.readthedocs.io/en/latest/tutorials/installation.html).

### Requiremenets
* Linux or macOS with Python ≥ 3.6
* PyTorch ≥ 1.8 and torchvision that matches the PyTorch installation. Install them together at pytorch.org to make sure of this
* fvcore
* pytorch_transformers
* jsonlines
* pycocotools

## Getting Started 
See [Getting Started with X-modaler](https://xmodaler.readthedocs.io/en/latest/tutorials/getting_started.html)

# Training with X-modaler

We provide a general training script, **`train_net.py`**, which can be used to train all configurations in **X-modaler**.  
You can either run it directly or use it as a reference to write your own customized training script.

---

## Environment Setup
You can follow the setup instructions in [**SCD_Net**](https://github.com/YehLi/xmodaler/tree/master) to construct your environment.

---

## Training

To train a model (e.g., [UpDown](https://drive.google.com/drive/folders/1vx9n7tAIt8su0y_3tsPJGvMPBMm8JLCZ), [CLIP-feature](https://github.com/jianjieluo/OpenAI-CLIP-Feature)) with **`train_net.py`**, first prepare the corresponding datasets as described in the dataset section, then run:

### 1. Teacher Forcing
```bash
python train_net.py --num-gpus 4 \
    --config-file configs/image_caption/scdnet/stage1/diffusion.yaml
```
### 2. Reinforcement Learning
```bash
python train_net.py --num-gpus 4 \
    --config-file configs/image_caption/scdnet/stage1/diffusion_rl.yaml
