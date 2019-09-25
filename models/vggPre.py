import keras
from keras import Input,Model
from keras.applications.vgg16 import VGG16
from keras.layers import Conv2D, MaxPooling2D,Dropout,concatenate,UpSampling2D

def vggPre(base_model):
    
    inputs = base_model.layers[0].output
    new_model = base_model

    up1 = concatenate([UpSampling2D((2,2))(new_model.layers[-1].output),new_model.layers[-2].output], axis=-1)
    conv8 = Conv2D(256,(3,3), activation='relu', padding='same')(up1)
    conv8 = Dropout(rate=0.2)(conv8)
    conv9 = Conv2D(256,(3,3), activation='relu', padding='same')(conv8)
    
    up2 = concatenate([UpSampling2D((2,2))(conv9),new_model.layers[-6].output], axis=-1)
    conv9 = Conv2D(128,(3,3), activation='relu', padding='same')(up2)
    conv9 = Dropout(rate=0.2)(conv9)
    
    conv10 = Conv2D(128,(3,3), activation='relu', padding='same')(conv9)
    conv10 = Dropout(rate=0.2)(conv10)
    
    up3 = concatenate([UpSampling2D((2,2))(conv10),new_model.layers[-9].output], axis=-1)
    conv11 = Conv2D(64,(3,3), activation='relu', padding='same')(up3)
    conv11 = Dropout(rate=0.2)(conv11)
    conv12 = Conv2D(64,(3,3), activation='relu', padding='same')(conv11)
    
    out = Conv2D( 1, (1, 1) , padding='same')(conv12)
    
    model = keras.Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss = 'binary_crossentropy', metrics=['accuracy',recall_m,precision_m,f1_m])
    
    return model
