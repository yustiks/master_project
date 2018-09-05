"""VGG19 model for Keras.
# Reference
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](
    https://arxiv.org/abs/1409.1556)
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from keras import applications
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import preprocess_input
#from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras import optimizers
from keras.models import Sequential, Model 
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import backend as k 
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model 
import numpy as np
import numpy
from keras.layers import Conv2D, MaxPooling2D

img_width, img_height = 160, 160
train_data_dir = "/scratch/ii1n17/big_data_1000/train/"
validation_data_dir = "/scratch/ii1n17/big_data_1000/valid/"
test_data_dir = "/scratch/ii1n17/big_data_1000/test"
#top_model_weights_path = '/scratch/ii1n17/weigths/fc_model.h5'


nb_train_samples = 30720#112000#33600 #30720 #4125
nb_validation_samples = 7680#32000#9600 #7680 #466 
batch_size = 96 #16
epochs = 50

vgg_model = applications.VGG16(weights = "imagenet", include_top=False, input_shape = (img_width, img_height, 3))
layer_dict = dict([(layer.name, layer) for layer in vgg_model.layers])

x = layer_dict['block2_pool'].output

# Freeze the layers which you don't want to train. Here I am freezing the first 5 layers.
#for layer in model.layers[:7]:
#    layer.trainable = False
#model.trainable = False


#Adding custom Layers 
#top_model = Sequential()
#top_model.add(Flatten(input_shape = model.output_shape[1:]))
#top_model.add(Dense(256, activation = 'relu'))
#top_model.add(Dropout(0.5))
#top_model.add(Dense(1, activation = 'sigmoid'))

# fully-trained weights for the last layer 
#top_model.load_weights(top_model_weights_part)

#model_final.add(top_model)

#for layer in model.layers[:25]:
#     layer.trainable = False 

#x = model.output
x = Conv2D(filters = 64, kernel_size = (3,3), activation = 'relu')(x)
x = MaxPooling2D(pool_size = (2,2))(x)
x = Flatten()(x)
#x = Dense(1024, activation="relu")(x)
x = Dense(256, activation = "relu")(x)
x = Dropout(0.5)(x)
#x = Dense(1024, activation="relu")(x)
predictions = Dense(2, activation="softmax")(x)

# creating the final model 
model_final = Model(input = vgg_model.input, output = predictions)

for layer in model_final.layers[:7]:
     layer.trainable = False 
# compile the model 
model_final.compile(loss = "categorical_crossentropy", 
#optimizer = optimizers.SGD(lr=0.0001, momentum=0.9), 
optimizer = 'rmsprop',
metrics=["accuracy"])

# Initiate the train and test generators with data Augumentation 
train_datagen = ImageDataGenerator(
#    rescale = 1./255
    preprocessing_function=preprocess_input
#    ,zoom_range = 0.2
)
test_datagen = ImageDataGenerator(
#    rescale = 1./255
    preprocessing_function=preprocess_input
#    ,zoom_range = 0.2
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_height, img_width),
    batch_size = batch_size, 
    class_mode = "categorical")

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_height, img_width),
    class_mode = "categorical",
    shuffle = False)

test_generator = test_datagen.flow_from_directory(
    test_data_dir, 
    target_size = (img_height, img_width),
    batch_size = batch_size,
    class_mode = "categorical",
    shuffle = False
)

# Save the model according to the conditions  
checkpoint = ModelCheckpoint("vgg19.hdf5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=2, mode='auto')


# Train the model 
history_tl = model_final.fit_generator(
    train_generator,
    samples_per_epoch = nb_train_samples,
#    steps_per_epoch = nb_train_samples//batch_size,
    epochs = 1,
    validation_data = validation_generator,
    nb_val_samples = nb_validation_samples,
#    validation_steps = nb_validation_samples//batch_size,
    callbacks = [checkpoint, early])

#model_final = load_model('/scratch/ii1n17/inceptionv3-ft.model')
test_steps_per_epoch = numpy.math.ceil(float(test_generator.samples)/test_generator.batch_size)
raw_predictions = model_final.predict_generator(test_generator, steps = test_steps_per_epoch)
predictions = numpy.argmax(raw_predictions, axis = 1)

print('Prediction distribution ' + str(numpy.bincount(predictions)))
print('Groundtruth distribution ' + str(numpy.bincount(test_generator.classes)))

from sklearn import metrics 
class_labels = [item[0] for item in sorted(test_generator.class_indices.items(), key = lambda x: x[1])]
print(metrics.classification_report(test_generator.classes, predictions, target_names=class_labels))

