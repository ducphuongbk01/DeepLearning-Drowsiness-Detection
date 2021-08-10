import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.applications.MobileNet()


for layer in model.layers:
    layer.trainable = False

base_input = model.layers[0].input
base_output = model.layers[-4].output

Flat_layer = layers.Flatten()(base_output)
final_output = layers.Dense(1)(Flat_layer)
final_output = layers.Activation('sigmoid')(final_output)

model = Model(inputs = base_input, outputs = final_output)

model.load_weights('./Model/Update_2/01.h5')

optim = tf.keras.optimizers.SGD(learning_rate = 0.0001, momentum = 0.9)
model.compile(optimizer = optim, loss = "binary_crossentropy", metrics = ['acc'])

train_folder = "./Data/Training"
valid_folder = "./Data/Validation"

train_datagen = ImageDataGenerator(rescale = 1.0/255.0,
                                    rotation_range = 45,
                                    width_shift_range = 0.2,
                                    height_shift_range = 0.2,
                                    shear_range = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True) 

valid_datagen = ImageDataGenerator(rescale = 1.0/255.0)

train_generator = train_datagen.flow_from_directory(train_folder, batch_size = 32, class_mode = 'binary', target_size = (224, 224))
valid_generator = train_datagen.flow_from_directory(valid_folder, batch_size = 32, class_mode = 'binary', target_size = (224, 224))

checkpoint_path = "./Model/Update_3/{epoch:02d}.h5"

cp_callback = tf.keras.callbacks.ModelCheckpoint(
                                                filepath=checkpoint_path, 
                                                verbose=1, 
                                                save_weights_only=True,
                                                save_freq = "epoch")

history = model.fit_generator(
                            train_generator,
                            validation_data = valid_generator,
                            steps_per_epoch = 100,
                            epochs = 20,
                            callbacks=[cp_callback],
                            validation_steps = 50,
                            verbose = 1)

print("Finished training!")