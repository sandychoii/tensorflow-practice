import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.layers import InputLayer, Conv2D, MaxPool2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy

## Data Preparation
dataset, dataset_info = tfds.load('malaria', with_info=True, as_supervised=True, split=['train'], shuffle_files=True)   

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
TEST_RATIO = 0.2

def splits(dataset, TRAIN_RATIO, VAL_RATIO, TEST_RATIO):
  DATASET_SIZE = len(dataset)
  train_dataset = dataset.take(int(TRAIN_RATIO*DATASET_SIZE))
  print(list(train_dataset.as_numpy_iterator()))

  val_test_dataset = dataset.skip(int(TRAIN_RATIO*DATASET_SIZE))
  val_dataset = val_test_dataset.take(int(VAL_RATIO*DATASET_SIZE))
  print(list(val_dataset.as_numpy_iterator()))

  test_dataset = val_test_dataset.skip(int(VAL_RATIO*DATASET_SIZE))
  print(list(test_dataset.as_numpy_iterator()))

  return train_dataset, val_dataset, test_dataset

train_dataset, val_dataset, test_dataset = splits(dataset[0], TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
print(list(train_dataset.take(1).as_numpy_iterator()),
      list(val_dataset.take(1).as_numpy_iterator()),
        list(test_dataset.take(1).as_numpy_iterator()))


## Data Preprocessing
IM_SIZE = 224
def resize_rescale(image, label):
  return tf.image.resize(image, (IM_SIZE, IM_SIZE))/255.0, label

train_dataset = train_dataset.map(resize_rescale)
val_dataset = val_dataset.map(resize_rescale)
test_dataset = test_dataset.map(resize_rescale)

BATCH_SIZE = 32
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

## Model
model = tf.keras.Sequential([InputLayer(input_shape=(IM_SIZE, IM_SIZE, 3)),
                             
                             Conv2D(filters=6, kernel_size=3, strides=1, padding='valid', activation='relu'),
                             BatchNormalization(),
                             MaxPool2D(pool_size=2, strides=2),

                             Conv2D(filters=16, kernel_size=3, strides=1, padding='valid', activation='relu'),
                             BatchNormalization(),
                             MaxPool2D(pool_size=2, strides=2),

                             Flatten(),
                             
                             Dense(100, activation='sigmoid'),
                             BatchNormalization(),
                             Dense(10, activation='sigmoid'),
                             BatchNormalization(),
                             Dense(1, activation='sigmoid')])
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss=BinaryCrossentropy(), 
              metrics=['accuracy'])

# Training
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, verbose=1)

# Evaluation
model.evaluate(test_dataset, verbose=1)

# Prediction
def parasite_or_not(x):
  ax = plt.subplot(3,3,i+1)
  if (x<0.5):
    return str('P')
  else:
    return str('U')
  
# for i, (image, label) in enumerate(test_dataset.take(9)):
#   ax = plt.subplot(3, 3, i+1)
#   plt.imshow(image[0])
#   plt.title(parasite_or_not(label.numpy()[0]) + ":"+ str(parasite_or_not(model.predict(image)[0][0])))
#   plt.axis("off")

# Save Model
model.save("maralaria_cnn")