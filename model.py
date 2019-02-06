import os
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard

class Model:
    def __init__(self):
        self.input_shape = (48, 48, 1)
        self.batch_size = 64
        self.epochs = 100
        self.verbose = 2
        self.tensorboard = TensorBoard(log_dir='logs/{}'.format(time.time()))
        self.lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=3, verbose=1)
        self.early_stopper = EarlyStopping(monitor='val_loss', min_delta=0, patience=8, verbose=1, mode='auto')

    def build_model(self):
        model = Sequential()

        model.add(Conv2D(
            32,
            kernel_size=(3, 3),
            activation='relu',
            input_shape=self.input_shape,
            kernel_regularizer = l2(0.01)
        ))

        model.add(Conv2D(
            64,
            (3, 3),
            activation='relu',
            padding='same'
        ))

        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(
            128,
            (3, 3),
            activation='relu',
            padding='same'
        ))

        model.add(Conv2D(
            128,
            (3, 3),
            activation='relu',
            padding='same'
        ))

        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))

        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.4))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(7, activation='softmax'))

        model.compile(
            optimizer=Adam(lr=0.001,beta_1=0.9, beta_2=0.999, epsilon=1e-7),
            metrics=['accuracy'],
            loss='categorical_crossentropy'
        )

        self.model = model

    def train(self, images, labels):
        self.history = self.model.fit(
            images, labels,
            batch_size=self.batch_size,
            verbose=self.verbose,
            epochs=self.epochs,
            callbacks=[self.tensorboard, self.lr_reducer, self.early_stopper],
            validation_split=0.15,
            shuffle=True
        )

        self.model.save(os.path.join(os.getcwd(), 'model', 'saved_models', 'emotion{}.h5'.format(time.time())))