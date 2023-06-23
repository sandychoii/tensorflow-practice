import tensorflow as tf
import pandas as pd
import seaborn as sns
from tensorflow.keras.layers import Normalization, Dense, InputLayer
from tensorflow.keras.losses import MeanSquaredError, Huber, MeanAbsoluteError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

# Data preparation
df = pd.read_csv("train.csv", delimiter=',')

tensor_data = tf.constant(df)
tensor_data = tf.cast(tensor_data, tf.float32)
tensor_data = tf.random.shuffle(tensor_data)

X = tensor_data[:, 3:-1] # using from years ~
y = tensor_data[:, -1] # (1000,)
y = tf.expand_dims(y, axis = -1) # (1000, 1)

### train-val-test split
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
DATASET_SIZE = len(X)

X_train = X[:int(DATASET_SIZE*TRAIN_RATIO)]
y_train = y[:int(DATASET_SIZE*TRAIN_RATIO)]

X_val= X[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]
y_val = y[int(DATASET_SIZE*TRAIN_RATIO):int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO))]

X_test = X[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]
y_test = y[int(DATASET_SIZE*(TRAIN_RATIO+VAL_RATIO)):]

print("Train data length", len(X_train))
print("Val data length", len(X_val))
print("Test data length", len(X_test))


### to Dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE) # prefetch: prepare next dataset while training the batch

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE) 

test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.shuffle(buffer_size=8, reshuffle_each_iteration=True).batch(32).prefetch(tf.data.AUTOTUNE) 

### Normalization
normalizer = Normalization()
normalizer.adapt(train_dataset)

# Model 
model = tf.keras.Sequential([
                              InputLayer(input_shape=(8,)),
                              normalizer,
                              Dense(128, activation="relu"),
                              Dense(128, activation="relu"),
                              Dense(128, activation="relu"),
                              Dense(1) # activation can interfere the way the model comes up with the outputs
                            ])
print(model.summary())

model.compile(optimizer=Adam(learning_rate=0.1),
              loss=MeanAbsoluteError(),
              metrics=RootMeanSquaredError())
history = model.fit(train_dataset, validation_data=val_dataset, epochs=100, verbose=1) 

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val_loss'])
plt.savefig('./result/model_loss.png')

plt.plot(history.history['root_mean_squared_error'])
plt.plot(history.history['val_root_mean_squared_error'])
plt.title('model performance')
plt.ylabel('rmse')
plt.xlabel('epoch')
plt.legend(['train', 'val'])
plt.savefig('./result/model_performance.png')

# Model Evaluation
model.evaluate(X_test,y_test)

# Model test
y_true = list(y_test[:,0].numpy())
y_pred = list(model.predict(X_test)[:,0])

ind = np.arange(100) # fixing position
plt.figure(figsize=(40,12))

width = 0.4

plt.bar(ind, y_pred, width, label='Predicted Car Price')
plt.bar(ind+width, y_true, width, label='Predicted Car Price')

plt.xlabel("Actual vs Predicted Prices")
plt.ylabel("Car Price Prices")

plt.savefig('./result/compare.png')
