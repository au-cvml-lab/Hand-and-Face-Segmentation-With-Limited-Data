import keras
from keras.layers import Conv2D, MaxPooling2D,Dropout,concatenate,UpSampling2D

def SNet(input_size = (400,400,3)):
    
    model = keras.Model()
    inputs = keras.Input(input_size)

    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Dropout(0.2)(conv3)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)

    up1 = concatenate([UpSampling2D((2, 2))(conv3), conv2], axis=-1)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(up1)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv4)

    up2 = concatenate([UpSampling2D((2, 2))(conv4), conv1], axis=-1)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(up2)
    conv5 = Dropout(0.2)(conv5)
    conv5 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv5)
    
    out = Conv2D( 1, (1, 1) , padding='same')(conv5)
    
    model = keras.Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer=keras.optimizers.Adam(1e-4), loss = 'binary_crossentropy', metrics=['accuracy'])
    # or you can use other metrics
    #model.compile(optimizer=keras.optimizers.Adam(1e-4), loss = 'binary_crossentropy', metrics=['accuracy','f1_m',....])
    
    return model
