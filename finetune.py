from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import numpy
from keras.models import Model
from keras.layers import Dense, Input, Flatten
from keras import optimizers

import numpy as np
import os
import time


from datetime import datetime, timedelta

def GetTime(ms):

    d = datetime.fromtimestamp(ms)
    print("HOURS:MIN:SEC")
    print("%d:%d:%d" % ( d.hour, d.minute, d.second))



# load data
train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#    rescale = 1./255,
#    fill_mode = "nearest",
#    width_shift_range = 0.2,
#    height_shift_range= 0.2,
#    rotation_range=30
)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input,
#    rescale = 1./255,
#    fill_mode = "nearest",
#    width_shift_range = 0.2,
#    height_shift_range= 0.2,
#    rotation_range=30
)

#train_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, 
#    rescale=1. / 255,
#    shear_range=0.1,
#    zoom_range=0.25,
#    rotation_range=45,
#    width_shift_range=0.25,
#    height_shift_range=0.25,
#    horizontal_flip=True,
#    channel_shift_range=0.07 
#)

#test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input, #rescale=1. / 255)

# the number of images that will be processed in a single step
batch_size=96
# the size of the images that we'll learn on - we'll use their natural size
image_size=(224, 224)

train_generator = train_datagen.flow_from_directory(
        '/scratch/ii1n17/data_1000_resnet/train',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical')

valid_generator = test_datagen.flow_from_directory(
        '/scratch/ii1n17/data_1000_resnet/valid',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

test_generator = test_datagen.flow_from_directory(
        '/scratch/ii1n17/data_1000_resnet/test',
        target_size=image_size,
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

num_classes = len(train_generator.class_indices)

def hack_resnet(input_size, num_classes):
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_size)
    x = base_model.output
    x = Flatten()(x)
    x = Dense(num_classes, activation='softmax', name='fc1000')(x)
    
    # this is the model we will train
    newmodel = Model(inputs=base_model.input, outputs=x)

    return newmodel

#--model = hack_resnet(train_generator.image_shape, num_classes)

# set weights in all but last layer
# to non-trainable (weights will not be updated)
#--for layer in model.layers[:len(model.layers)-2]:
#--    layer.trainable = False

# compile the model with a SGD/momentum optimizer
# and a very slow learning rate.
#--model.compile(loss='categorical_crossentropy',
#              optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0),
#--              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
#lr=0.002
#--              metrics=['accuracy'])

model = ResNet50(input_shape=image_size, include_top=True,weights='imagenet')
last_layer = model.get_layer('avg_pool').output
x= Flatten(name='flatten')(last_layer)
out = Dense(num_classes, activation='softmax', name='output_layer')(x)
custom_resnet_model = Model(inputs=model.input,outputs= out)


for layer in custom_resnet_model.layers[:-2]:
	layer.trainable = False

custom_resnet_model.layers[-1].trainable

custom_resnet_model.compile(loss='categorical_crossentropy',
#optimizer='adam',
optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
metrics=['accuracy'])


t=time.time()
GetTime(t)

nb_train_samples = 33600 #30720 #10240 50807 #
nb_validation_samples = 5600 #7680 #2560 12701 #

# Fit the model
model.fit_generator(
        train_generator,
        steps_per_epoch= nb_train_samples//batch_size, 
        validation_data=valid_generator,
        validation_steps= nb_validation_samples//batch_size,
        epochs=1,
        verbose=2)

t1 = time.time()
GetTime(t1)
print('Training time: %s sec' % (t1 - t))

model.save('/scratch/ii1n17/ResNetModel/resnet_8aug.h5')
# Final evaluation of the model
test_steps_per_epoch = numpy.math.ceil(float(test_generator.samples) / test_generator.batch_size)
raw_predictions = model.predict_generator(test_generator, steps=test_steps_per_epoch)
predictions = numpy.argmax(raw_predictions, axis=1)

print("Prediction Distribution:  " + str(numpy.bincount(predictions)))
print("Groundtruth Distribution: " + str(numpy.bincount(test_generator.classes)))

from sklearn import metrics
class_labels = [item[0] for item in sorted(test_generator.class_indices.items(), key=lambda x: x[1])] #get a list of classes
print(metrics.classification_report(test_generator.classes, predictions, target_names=class_labels))

