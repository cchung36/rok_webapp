import os, sys
from mnist import MNIST
sys.path.insert(0,os.path.abspath(os.path.join(os.path.dirname(__file__),"../model/")))
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from model import CNN

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
  except RuntimeError as e:
    print(e)

dir='/src/dataset'
mndata=MNIST(dir)

train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()

trainX = tf.convert_to_tensor(train_images,dtype=tf.float32)
trainY = tf.convert_to_tensor(train_labels,dtype=tf.uint32)
testX = tf.convert_to_tensor(test_images,dtype=tf.float32)
testY = tf.convert_to_tensor(test_labels,dtype=tf.uint32)

trainX = trainX/255.0
testX = testX/255.0

trainX = tf.reshape(trainX,[trainX.shape[0],28,28,1])
testX = tf.reshape(testX,[testX.shape[0],28,28,1])

print('shape1:{}'.format(trainX.shape))
print('shape2:{}'.format(testY.shape))

num_classes = 10

trainY = to_categorical(trainY)
testY  = to_categorical(testY)

# Fit the model
checkpoint_path = "/src/checkpoint/model-{epoch:04d}.tf"
checkpoint_dir = os.path.dirname(checkpoint_path)

cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    monitor='val_accuracy',
#	save_weights_only=True,
    mode='max',
    save_best_only=True)

# build the model
raw_input=(28,28,1)
model=CNN()
model.build((None,*raw_input))
model.build_graph(raw_input).summary()

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))

model.fit(trainX, trainY, validation_data=(testX, testY), epochs=1875, batch_size=32, callbacks=[cp_callback], verbose=2)
# Final evaluation of the model
scores = model.evaluate(testX, testY, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))