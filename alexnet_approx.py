import tensorflow as tf
import os
import time
import tensorflow_datasets as tfds
import sys
import keras
from python.keras.layers.am_convolutional import AMConv2D
from python.keras.layers.amdenselayer import denseam

# To ensure all training runs get the same starting point
tf.random.set_seed(
    20230414
)

#(ds_train, ds_test), ds_info = tfds.load(
 #       'cifar10',
  #      split=['train', 'test'],
  #      shuffle_files = True,
  #      as_supervised = True,
  #      with_info = True,
  #      )

# Load CIFAR10 Dataset
(train_images, train_labels), (test_images, test_labels) = keras.datasets.cifar10.load_data()

# Set class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 227x227
    image = tf.image.resize(image, (227,227))
    return image, label

lut_file = './lut/MBM_1.bin'

root_logdir = os.path.join(os.curdir, "logs/fit")
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

validation_images, validation_labels = train_images[:5000], train_labels[:5000]
train_images, train_labels = train_images[5000:], train_labels[5000:]

train_ds = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
validation_ds = tf.data.Dataset.from_tensor_slices((validation_images, validation_labels))

train_ds_size = tf.data.experimental.cardinality(train_ds).numpy()
test_ds_size = tf.data.experimental.cardinality(test_ds).numpy()
validation_ds_size = tf.data.experimental.cardinality(validation_ds).numpy()
print("Training data size:", train_ds_size)
print("Test data size:", test_ds_size)
print("Validation data size:", validation_ds_size)

train_ds = (train_ds\
                  .map(process_images)
                  .shuffle(buffer_size=6000)
                  .batch(batch_size=32, drop_remainder=True))

test_ds = (test_ds\
                  .map(process_images)
                  .shuffle(buffer_size=6000)
                  .batch(batch_size=32, drop_remainder=True))

validation_ds = (validation_ds\
                  .map(process_images)
                  .shuffle(buffer_size=6000)
                  .batch(batch_size=32, drop_remainder=True))

model = keras.models.Sequential([
    AMConv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,3), mant_mul_lut=lut_file),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    AMConv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same", mant_mul_lut=lut_file),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    AMConv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", mant_mul_lut=lut_file),
    keras.layers.BatchNormalization(),
    AMConv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", mant_mul_lut=lut_file),
    keras.layers.BatchNormalization(),
    AMConv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same", mant_mul_lut=lut_file),
    keras.layers.BatchNormalization(),
    keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
    keras.layers.Flatten(),
    denseam(4096, activation='relu', mant_mul_lut=lut_file),
    keras.layers.Dropout(0.5),
    denseam(4096, activation='relu', mant_mul_lut=lut_file),
    keras.layers.Dropout(0.5),
    denseam(10, activation='softmax', mant_mul_lut=lut_file)
])

model.compile(
    optimizer=tf.keras.optimizers.SGD(0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
    )

model.fit(
    train_ds,
    epochs=50,
    validation_data = validation_ds,
    validation_freq = 1,
    callbacks=[tensorboard_cb]
)

model.evaluate(test_ds)
