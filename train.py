# -*- coding: utf-8 -*-
"""
Created on Tue Oct 19 19:20:53 2021
paper : https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf

@author: Anurag
"""

import os
# import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
import tensorflow as tf

# GPU memory use growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

pos_path = os.path.join('data', 'positive')
neg_path = os.path.join('data', 'negative')
anc_path = os.path.join('data', 'anchor')

# load images
anchor = tf.data.Dataset.list_files(anc_path + '\*.jpg').take(300)
positive = tf.data.Dataset.list_files(pos_path + '\*.jpg').take(300)
negative = tf.data.Dataset.list_files(neg_path + '\*.jpg').take(300)
dir_test = anchor.as_numpy_iterator()
print(dir_test.next())


def pre_process(file_path):
    byte_img = tf.io.read_file(file_path)
    img = tf.io.decode_jpeg(byte_img)
    img = tf.image.resize(img, (100, 100))
    img = img / 255.0
    return img


def preproces_twin(in_img, val_img, label):
    return (pre_process(in_img), pre_process(val_img), label)


# img = pre_process('data\\anchor\\24c33e30-3104-11ec-85c8-04d4c47a9fee.jpg')
# plt.imshow(img)
# plt.show()


positive = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negative = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data = positive.concatenate(negative)
# print(len(positive))
# print('------------------------------------------------------')
# print(len(data))
# samples = data.as_numpy_iterator()
# ex = samples.next()
# res = preproces_twin(*ex)


data = data.map(preproces_twin)
data = data.cache()
data = data.shuffle(buffer_size=1024)

### check data creation 
# sample = data.as_numpy_iterator()
# samp = sample.next()
# plt.imshow(samp[0])
# plt.imshow(samp[1])
# plt.show()
# print(samp[2])

# train and test split
train_data = data.take(round(len(data) * .7))  # taking 70% of data
train_data = train_data.batch(16)  # data in batch of 16 images
train_data = train_data.prefetch(8)

test_data = data.skip(round(len(data) * .7))
test_data = test_data.take(round(len(data) * .3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)


def embedding_layer():
    inp = Input(shape=(100, 100, 3), name='input_image')

    # first block
    c1 = Conv2D(64, (10, 10), activation='relu')(inp)
    m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

    # second block
    c2 = Conv2D(128, (7, 7), activation='relu')(m1)
    m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

    # third block
    c3 = Conv2D(128, (4, 4), activation='relu')(m2)
    m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

    # final embedding block
    c4 = Conv2D(256, (4, 4), activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096, activation=('sigmoid'))(f1)

    return Model(inputs=[inp], outputs=[d1], name='embedding')


embeding = embedding_layer()
print(embeding.summary())


class L1_dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, input_embed, val_embed):
        return tf.math.abs(input_embed - val_embed)


def make_siamese():
    input_img = Input(shape=(100, 100, 3), name='input_img')

    val_img = Input(shape=(100, 100, 3), name='validation_img')

    siamese_layer = L1_dist()
    siamese_layer._name = 'distance'
    distances = siamese_layer(embeding(input_img), embeding(val_img))

    # classification layer
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_img, val_img], outputs=classifier, name='SiameseNetwork')


siamese_model = make_siamese()
print(siamese_model.summary())

# training netwrok

binary_cross_loss = tf.losses.BinaryCrossentropy()

optimizer_adam = tf.keras.optimizers.Adam(0.0001)  # 1e-4

# checkpoints
chechkpoit_dir = './check_point'
chechkpoit_prefix = os.path.join(chechkpoit_dir, 'ckpt')
chechkpoit = tf.train.Checkpoint(opt=optimizer_adam, siamese_model=siamese_model)


@tf.function
def train_step(batch):
    with tf.GradientTape() as tape:
        # get anchor
        X = batch[:2]
        y = batch[2]

        # forward pass
        yhat = siamese_model(X, training=True)

        # calculate loss
        loss = binary_cross_loss(y, yhat)

    # calculate gradient
    grad = tape.gradient(loss, sources=siamese_model.trainable_variables)

    # calculate updated weights and aply to model
    optimizer_adam.apply_gradients(zip(grad, siamese_model.trainable_variables))

    return loss


def train(data, EPOCHS):
    # loop through epochs
    for epochs in range(1, EPOCHS + 1):
        print(f'Epoch {epochs}/{EPOCHS}')
        progbar = tf.keras.utils.Progbar(len(data))

        # loop through each batch
        for idx, batch in enumerate(data):
            # run train step here
            train_step(batch)
            progbar.update(idx + 1)

        # save checkpoint
        if epochs % 10 == 0:
            chechkpoit.save(file_prefix=chechkpoit_prefix)


EPOCHS = 50
train(train_data, EPOCHS)

siamese_model.save('siamesemodel.h5')


