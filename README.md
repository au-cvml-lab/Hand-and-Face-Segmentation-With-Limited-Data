# Hands-and-Face-Segmentation
Hands and Face Segmentation with Deep Convolutional Networks using limited labeled data.



## Usage

```python
from models import SNet, vggPre
from utils import *

model = SNet() # For using unet based architecture
# model = vggPre() # For using VGG-16 based pretrained network

# Metrics for evaluting models, if you want to use pretrained model.
dependencies = {
	'f1_m' = f1_m,
	'recall_m' = recall_m,
	'precision_m' = precision_m
}

```

## DATASET

Ankara University Computer Vision & Machine Learning Labaratory (CVML LAB) Turkish Sign Language (TSL) Dataset.

### Pre-processing



## Models architectures

Mode1-1

![SegNetwork-VGG](https://user-images.githubusercontent.com/23141486/65586920-a7328700-df8d-11e9-9c79-009eba1c5592.jpg)




