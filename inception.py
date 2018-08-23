#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import glob
import argparse
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from keras import backend as K
from keras import __version__
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.models import Model
from keras.layers import Dense, AveragePooling2D, GlobalAveragePooling2D, Input, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Sequential
from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TensorBoard


NB_EPOCHS = 1
BAT_SIZE = 96
nclass = 2
#NB_IV3_LAYERS_TO_FREEZE = 172


def get_nb_files(directory):
    """Get number of files by searching directory recursively"""
    if not os.path.exists(directory):
        return 0
    cnt = 0
    for r, dirs, files in os.walk(directory):
        for dr in dirs:
            cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def setup_to_transfer_learn(model, base_model):
    """Freeze all layers and compile the model"""
    for layer in base_model.layers:
        layer.trainable = False
    model.compile(optimizer='rmsprop',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])


def add_new_last_layer(base_model, nb_classes):
    """Add last layer to the convnet
    
    Args:
        base_model: keras model excluding top
        nb_classes: # of classes
        
    Returns:
        new keras model with last layer
    """
    x = base_model.output
    x = AveragePooling2D((8, 8), border_mode='valid', name='avg_pool')(x)
    x = Dropout(0.4)(x)
    x = Flatten()(x)
    predictions = Dense(2, activation='softmax')(x)
    model = Model(input=base_model.input, output=predictions)
    return model
    
"""
def setup_to_finetune(model):
  Freeze the bottom NB_IV3_LAYERS and retrain the remaining top layers.
  note: NB_IV3_LAYERS corresponds to the top 2 inception blocks in the inceptionv3 arch
  Args:
    model: keras model
  
  for layer in model.layers[:NB_IV3_LAYERS_TO_FREEZE]:
     layer.trainable = False
  for layer in model.layers[NB_IV3_LAYERS_TO_FREEZE:]:
     layer.trainable = True
  model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
"""

def train(args):
    """Use transfer learning and fine-tuning to train a network on a new dataset"""
    train_img = '/scratch/ii1n17/data_1000_resnet/train/' 
    validation_img = '/scratch/ii1n17/data_1000_resnet/valid/'
    nb_epoch = int(args.nb_epoch)
    nb_train_samples = get_nb_files(train_img)
    nb_classes = len(glob.glob(train_img + "/*"))
    # data prep
    train_datagen = ImageDataGenerator(
			preprocessing_function=preprocess_input
)

    validation_datagen = ImageDataGenerator(
			preprocessing_function=preprocess_input
)
    
    train_generator = train_datagen.flow_from_directory(
			train_img,
			target_size=(139, 139),
			batch_size=BAT_SIZE,
			class_mode='categorical'
			)
    validation_generator = validation_datagen.flow_from_directory(
			validation_img,
			target_size=(139, 139),
			batch_size=BAT_SIZE,
			class_mode='categorical'
			)
    if(K.image_dim_ordering() == 'th'):
        input_tensor = Input(shape=(3, 139, 139))
    else:
        input_tensor = Input(shape=(139, 139, 3))
    
    # setup model
    base_model = InceptionV3(input_tensor = input_tensor,weights='imagenet', include_top=False) #include_top=False excludes final FC layer
    base_model.trainable = False

    add_model = Sequential()
    add_model.add(base_model)
    add_model.add(GlobalAveragePooling2D())
    add_model.add(Dropout(0.5))
    add_model.add(Dense(nclass, 
                    activation='softmax'))

    model = add_model
    model.compile(loss='categorical_crossentropy', 
              optimizer=optimizers.SGD(lr=1e-4, 
                                       momentum=0.9),
              metrics=['accuracy'])
    model.summary()

   
    
    nb_train_samples = 33600 #30720 #4125
    nb_validation_samples = 9600 #7680 #466 

    file_path="inception.hdf5"

    checkpoint = ModelCheckpoint(file_path, monitor='acc', verbose=1, save_best_only=True, mode='max')

    early = EarlyStopping(monitor="acc", mode="max", patience=15)

    callbacks_list = [checkpoint, early] #early

#    history = model.fit_generator(train_gen, 
#                              epochs=2, 
#                              shuffle=True, 
#                              verbose=True,
#                              callbacks=callbacks_list)
    
    history_tl = model.fit_generator(train_generator, 
				    steps_per_epoch=nb_train_samples//BAT_SIZE, 
				    epochs=1, 
				    verbose=1,  
				    validation_data=validation_generator, 
				    validation_steps=nb_validation_samples//BAT_SIZE,  
				    callbacks=callbacks_list)
#fit_generator(train_generator,
#				    samples_per_epoch = BAT_SIZE,
#				    epochs = epochs,
#				    validation_data = validation_generator,
#				    nb_val_samples = nb_validation_samples,
#				    callbacks = [checkpoint, early]
#                                   samples_per_epoch=320,
#                                   nb_epoch=nb_epoch,
#                                   nb_val_samples=64
#) 
    model.save(args.output_model_file)
#    if args.plot:
    plot_training(history_tl)
        
        
def plot_training(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'r.')
    plt.plot(epochs, val_acc, 'r')
    plt.title('Training and validation accuracy')
    plt.savefig('/scratch/ii1n17/accuracy.png')
    print('pic saved ok')
    
    plt.figure()
    plt.plot(epochs, loss, 'r.')
    plt.plot(epochs, val_loss, 'r-')
    plt.title('Training and validation loss')
    plt.savefig('/scratch/ii1n17/loss.png')
    print('pic saved ok')
  

  




if __name__=="__main__":
    
    
    a = argparse.ArgumentParser()
    a.add_argument("--nb_epoch", default=NB_EPOCHS)
    a.add_argument("--batch_size", default=BAT_SIZE)
    a.add_argument("--plot", action="store_true")
    a.add_argument("--output_model_file", default="inceptionv3-ft.model")
    args = a.parse_args()
    
    train(args)
