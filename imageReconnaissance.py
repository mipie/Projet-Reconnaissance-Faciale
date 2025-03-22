from tensorflow.python.keras.layers import Input, Dense, Layer, Conv2D, MaxPooling2D, Flatten
from tensorflow.python.keras.models import Model, load_model
from tensorflow.python.keras.losses import BinaryCrossentropy
from tensorflow.python.keras.optimizer_v2 import adam
from tensorflow.python.keras.utils.all_utils import Progbar

import tensorflow as tf
from os.path import exists
import cv2
import os
import random
import numpy as np
from matplotlib import pyplot as plt




# Avoid OOM errors by setting GPU Memory consumption Growth
gpus = tf.config.experimental.list_logical_devices("GPU")
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Setup paths
POS_PATH = os.path.join("data", "positive")
NEG_PATH = os.path.join("data", "negative")
ANC_PATH = os.path.join("data", "achor")


paths = [POS_PATH, NEG_PATH, ANC_PATH]

# Make the directories
for path in paths:
    if (not exists(path)):
        os.makedirs(path)


#Move LFW Images to the following repository data/negative
for directory in os.listdir("lfw"):
    for file in os.listdir(os.path.join("lfw", directory)):
        EX_PATH = os.path.join("lfw", directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

# Get image directory
anchor = tf.data.Dataset.list_files(ANC_PATH+"\*.jpg").take(100)
negative = tf.data.Dataset.list_files(NEG_PATH+"\*.jpg").take(100)
positive = tf.data.Dataset.list_files(POS_PATH+"\*.jpg").take(100)

#dir_test = negative.as_numpy_iterator()

#print(dir_test.next())


def preprocess(file_path):

    #Read in image in the file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)
    #Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100,100))
    #Scale image to be between 0 and 1 
    img = img / 255.0
    return img

#plt.imshow(img)
#plt.show()

#  This methods help us to combine the independent features and their target as one dataset. (from_tensor_slices)
positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positives.concatenate(negatives)

def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# Build dataloader pipeline
data = data.map(preprocess_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

# Trainning partition
train_data = data.take(round(len(data) * 0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# Test partition
test_data = data.skip(round(len(data) * 0.7))
test_data = test_data.take(round(len(data) * 0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)

# Build embendding layout
def make_embedding():
    inp = Input(shape=(100,100,3), name='input_image')
    
    # First block
    c1 = Conv2D(64,(10,10), activation='relu')(inp)
    m1 = MaxPooling2D(64,(2,2), padding='same')(c1)

    # Second block
    c2 = Conv2D(64,(7,7), activation='relu')(m1)
    m2 = MaxPooling2D(64,(2,2), padding='same')(c2)

    # Third block
    c3 = Conv2D(64,(4,4), activation='relu')(m2)
    m3 = MaxPooling2D(64,(2,2), padding='same')(c3)

    # Final embedding block
    c4 = Conv2D(64,(10,10), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation='sigmoid')(f1)
    

    return Model(inputs=[inp], outputs=[d1], name='embedding')

# Build Distance Layer
embedding = make_embedding()
# Siamese L1 Distance class
class L1Dist(Layer):

    def __init__(self, **kwargs):
        super().__init__()
    
    # Magic happens here
    def call(self, input_embendding, validation_embedding):
        return tf.math.abs(input_embendding - validation_embedding)

def make_siamese_model():
    

    # Anchor image input in the network
    input_image = Input(name='input_img', shape=(100,100,3))

    # Validation image in the network
    valiation_image = Input(name='validation_img', shape=(100,100,3))

    # combine siamese distance components
    siamese_layer = L1Dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embedding(input_image), embedding(valiation_image))

    # Classification Layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, valiation_image], outputs=classifier, name='siameseNetwork')

siamese_model = make_siamese_model()

binary_cross_loss = BinaryCrossentropy()
optimizer = adam.Adam(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
checkpoint = tf.train.Checkpoint(opt=optimizer, siamese_model=siamese_model)

@tf.function
def train_step(batch):

    # Record all of our operations
    with tf.GradientTape() as tape:
        # Get anchor and positive/negative image
        x = batch[:2]
        # Get label
        y = batch[2]

        # Forward pass
        y_true = siamese_model(x, training=True)
        # Calculate loss
        loss = binary_cross_loss(y, y_true)

    # Calculate gradients
    grad = tape.gradient(loss, siamese_model.trainable_variables)

    # Calculate update weights and apply to siamese model
    optimizer.apply_gradients(zip(grad, siamese_model.trainable_variables))

    # Return loss
    return loss

def train(data, EPOCHS):
    # Loop through epochs
    for epoch in range(1, EPOCHS+1):
        print('\n Epoch {}/{}'.format(epoch,EPOCHS))
        progbar = Progbar(len(data))

        # Loop through each batch
        for idx, batch in enumerate(data):
            # Run train step here
            train_step(batch)
            progbar.update(idx+1)

        # Save checkpoints
        if epoch % 10 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

EPOCHS = 50
train(train_data,EPOCHS)

# Import metric calculations
from tensorflow.python.keras.metrics import Precision, Recall

#  Get a batch of test data
test_input, test_val, y_true = test_data.as_numpy_iterator().next()

# Make predictions
y_hat = siamese_model.predict([test_input, test_val])

# Post processing the results
[1 if prediction > 0.5 else 0 for prediction in y_hat]

# Creating a metric object
m = Recall()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
result = m.result()
array = np.array(result)


# Creating a metric object
m = Precision()

# Calculating the recall value
m.update_state(y_true, y_hat)

# Return Recall Result
result = m.result()
array = np.array(result)

# Set plot size
plt.figure(figsize=(10,8))

# SET FIRST SUBPLOT
plt.subplot(1,2,1)
plt.imshow(test_input[0])

# SET SECOND SUBPLOT
plt.subplot(1,2,2)
plt.subplot(test_val[0])

# RENDERS THE IMAGE
plt.show()

# Save weights
siamese_model.save('siamesemodel.h5')

# Reload model
model = load_model('siamesemodel.h5',
                   custom_objects={'L1Dist:': L1Dist, 'BynaryCrossentropy': BinaryCrossentropy})



# Make prediction with reloaded model
model.predict([test_input, test_val])
# View model summary
model.summary()