# Hands-and-Face-Segmentation
Hands and Face Segmentation with Deep Convolutional Networks using limited labeled data.



## Usage

Training.

```python
from models import model1, model2
from keras import Model
import utils
from keras.applications.vgg16 import VGG16

modelSNet = model1.SNet() # For using unet based architecture

# Using vgg based model.
modelVgg = VGG16(weights="imagenet", include_top=False, input_shape=(400,400,3))
base_model = Model(inputs=model.layers[0].output, outputs=model.layers[10].output)

# Freeze vgg layers.
for layer in base_model.layers:
    layer.trainable = False
    
modelVGG = model2.vggPre(base_model)

modelVGG.fit(...)
modelSNet.fit(...)



# Metrics for evaluting models, if you want to use pretrained model.
dependencies = {
	'f1_m' = utils.f1_m,
	'recall_m' = utils.recall_m,
	'precision_m' = utils.precision_m
}



```

## DATASET

Ankara University Computer Vision & Machine Learning Labaratory (CVML LAB) Turkish Sign Language (TSL) Dataset.

Models architectures

Model-1

![snet](https://user-images.githubusercontent.com/23141486/65591140-9c2f2500-df94-11e9-83bd-a02072e7ae88.jpg)

Model-2

![SegNetwork-VGG](https://user-images.githubusercontent.com/23141486/65586920-a7328700-df8d-11e9-9c79-009eba1c5592.jpg)
