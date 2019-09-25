import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D,Dropout,concatenate,UpSampling2D

def SNet(input_size = (400,400,3)):
    
    model = tf.keras.Model()
    inputs = tf.keras.Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same' )(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = SpatialDropout2D(rate=0.4)(conv3)
    conv3 = Conv2D(128, (3,3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    center = Conv2D(256, (5, 5), activation='relu', padding='same')(pool3)
    center = SpatialDropout2D(rate=0.5)(center)
    center = Conv2D(256, (5, 5), activation='relu', padding='same')(center)

    up1 = concatenate([UpSampling2D((2, 2))(center), conv3], axis=-1)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(up1)
    conv4 = SpatialDropout2D(rate=0.4)(conv4)
    conv4 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv2], axis=-1)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv5)

    up3 = concatenate([UpSampling2D((2, 2))(conv5), conv1], axis=-1)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(up3)
    conv6 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv6)
    
    out = Conv2D( 1, (1, 1) , activation='sigmoid')(conv6)
    
    model = tf.keras.Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss = 'binary_crossentropy', metrics=['accuracy'])
    # or you can use other metrics
    #model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss = 'binary_crossentropy', metrics=['accuracy','f1_m',....])
    
    return model
