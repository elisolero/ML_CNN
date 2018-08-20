from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# init CNN
classifier = Sequential()


#convolutional step 1
classifier.add(Conv2D(32,3,3, input_shape=(64,64,3), activation='relu'))
#poolong step 2
classifier.add(MaxPooling2D(pool_size=(2,2)))

# #convolutional step 1
# classifier.add(Conv2D(32,3,3, activation='relu'))
# #poolong step 2
# classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flatten step3
classifier.add(Flatten())

# ANN full connection step 4
classifier.add(Dense(output_dim=128 , activation='relu'))
classifier.add(Dense(output_dim=1 , activation='sigmoid'))


classifier.compile(optimizer='adam',loss='binary_crossentropy' , metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator
#datagenarator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

classifier.fit_generator(
    training_set,
        steps_per_epoch=8000,#number of images
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)