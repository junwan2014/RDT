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

### Training & Evaluation in Command Line

We provide a script in "train_net.py", that is made to train all the configs provided in X-modaler. You may want to use it as a reference to write your own training script.

To train a model(e.g., [UpDown](https://drive.google.com/drive/folders/1vx9n7tAIt8su0y_3tsPJGvMPBMm8JLCZ),  [CLIP-feature](https://github.com/jianjieluo/OpenAI-CLIP-Feature) with "train_net.py", first setup the corresponding datasets following datasets, then run:
