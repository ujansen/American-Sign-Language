from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
# Initialising the CNN
classifier = Sequential()

#First Layer
classifier.add(Conv2D(filters = 64, kernel_size = (4, 4), 
                      input_shape = (64, 64, 3), activation = 'relu'))

classifier.add(Conv2D(filters = 64, kernel_size = (4, 4), strides = 2,  activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(BatchNormalization(axis = 3, momentum = 0.8))

#Second Layer
classifier.add(Conv2D(filters = 128, kernel_size = (4, 4), activation = 'relu'))

classifier.add(Conv2D(filters = 128, kernel_size = (4, 4), strides = 2,  activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(BatchNormalization(axis = 3, momentum = 0.8))

#Third Layer

classifier.add(Conv2D(filters = 256, kernel_size = (4, 4), activation = 'relu'))

classifier.add(Conv2D(filters = 256, kernel_size = (4, 4), strides = 2,  activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(BatchNormalization(axis = 3, momentum = 0.8))

# Flattening
classifier.add(Flatten())

classifier.add(Dropout(0.5))

# Hidden Layer and Output Layer
classifier.add(Dense(units = 512, activation = 'relu'))
classifier.add(Dense(units = 24, activation = 'softmax'))

#Compiling the CNN
classifier.compile(optimizer= 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(rescale = 1./255, 
                             shear_range = 0.2,
                             validation_split = 0.1, 
                             zoom_range = 0.2,  
                             horizontal_flip = True,
                             samplewise_center = True, 
                             samplewise_std_normalization = True)

training_set = datagen.flow_from_directory('asl_alphabet_train', 
                                           target_size = (64, 64), 
                                           batch_size = 64,
                                           class_mode = 'categorical',
                                           subset = 'training')

test_set = datagen.flow_from_directory('asl_alphabet_train',
                                       target_size = (64, 64), 
                                       batch_size = 64,
                                       class_mode = 'categorical', 
                                       subset = 'validation')


classifier.fit_generator(training_set, steps_per_epoch = 1013, epochs = 20,
                         validation_data = test_set, validation_steps = 113)
