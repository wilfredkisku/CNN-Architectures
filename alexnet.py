import os

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model

inputs = Input(shape=(224, 224, 3))
x = Conv2D(filters=96, kernel_size=11, strides=4, padding='same', activation='relu')(inputs)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=3, strides=2)(x)
x = Conv2D(filters=256, kernel_size=5, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=3, strides=2)(x)
x = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=384, kernel_size=3, padding='same', activation='relu')(x)
x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=3, strides=2)(x)
x = Flatten()(x)
x = Dense(units=4096, activation='relu')(x)
x = Dense(units=4096, activation='relu')(x)
x = Dropout(rate=0.5)(x)
outputs = Dense(units=1000, activation='softmax')(x)


model = Model(inputs = inputs, outputs=outputs)

plot_model(model, show_shapes=True)
