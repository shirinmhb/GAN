from __future__ import print_function, division
import os
from tensorflow.keras.preprocessing import image
from keras.preprocessing.image import save_img
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
import matplotlib.pyplot as plt
import numpy as np
import cv2
size_img = 256

def read_samples():
  x = np.zeros(shape=(size_img, size_img, 1))
  train_covid = os.listdir('COVID,Non COVID-CT Images/train/COVID')
  train_not_covid = os.listdir('COVID,Non COVID-CT Images/train/Non-COVID')
  test_covid = os.listdir('COVID,Non COVID-CT Images/test/COVID')
  test_not_covid = os.listdir('COVID,Non COVID-CT Images/test/Non-COVID')
  test_covid = test_covid[:1]
  test_not_covid = test_not_covid[:1]
  num_train_covid, num_train_not_covid, num_test_covid, num_test_not_covid = len(train_covid), len(train_not_covid), len(test_covid), len(test_not_covid)
  train_shape = (num_train_covid+num_train_not_covid, x.shape[0], x.shape[1], x.shape[2])
  xTrain, yTrain = np.zeros(shape=(train_shape)), np.zeros(shape=(num_train_covid+num_train_not_covid))
  test_shape = (num_test_covid+num_test_not_covid, x.shape[0], x.shape[1], x.shape[2])
  xTest, yTest = np.zeros(shape=(test_shape)), np.zeros(shape=(num_test_covid+num_test_not_covid))
  for i in range(len(train_covid)):
    path_img = train_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/train/COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTrain[i] = x
    yTrain[i] = 1

  for i in range(len(train_not_covid)):
    path_img = train_not_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/train/Non-COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTrain[i+num_train_covid] = x
    yTrain[i+num_train_covid] = 0
  
  for i in range(len(test_covid)):
    path_img = test_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/test/COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTest[i] = x
    yTest[i] = 1

  for i in range(len(test_not_covid)):
    path_img = test_not_covid[i]
    img = image.load_img('COVID,Non COVID-CT Images/test/Non-COVID/'+path_img, target_size=(size_img, size_img), color_mode='grayscale')
    x = image.img_to_array(img)
    xTest[i+num_test_covid] = x
    yTest[i+num_test_covid] = 0

  print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
  return ((xTrain, yTrain), (xTest, yTest))

class ACGAN():
  def __init__(self):
    self.img_rows = size_img
    self.img_cols = size_img
    self.channels = 1
    self.img_shape = (self.img_rows, self.img_cols, self.channels)
    self.num_classes = 2
    self.latent_dim = 100
    losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(loss=losses, optimizer='adam', metrics=['accuracy'])
    self.generator = self.build_generator()
    noise = Input(shape=(self.latent_dim,))
    label = Input(shape=(1,))
    img = self.generator([noise, label])
    self.discriminator.trainable = False
    valid, target_label = self.discriminator(img)
    self.combined = Model([noise, label], [valid, target_label])
    self.combined.compile(loss=losses, optimizer='adam')

  def build_generator(self):
    x = int(size_img/4)
    model = Sequential()
    model.add(Dense(128 * x * x, activation="relu", input_dim=self.latent_dim))
    model.add(Reshape((x, x, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(self.channels, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))
    model.summary()
    noise = Input(shape=(self.latent_dim,))
    label = Input(shape=(1,), dtype='int32')
    label_embedding = Flatten()(Embedding(self.num_classes, self.latent_dim)(label))
    model_input = multiply([noise, label_embedding])
    img = model(model_input)
    return Model([noise, label], img)

  def build_discriminator(self):
    model = Sequential()
    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.summary()
    img = Input(shape=self.img_shape)
    features = model(img)
    validity = Dense(1, activation="sigmoid")(features)
    label = Dense(self.num_classes, activation="softmax")(features)
    return Model(img, [validity, label])

  def train(self, epochs, batch_size=128, sample_interval=50):
    (X_train, y_train), (_, _) = read_samples()
    y_train = y_train.astype('uint8')
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    y_train = y_train.reshape(-1, 1)
    print(X_train.shape, y_train.shape)
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):
      idx = np.random.randint(0, X_train.shape[0], batch_size)
      imgs = X_train[idx]
      noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
      sampled_labels = np.random.randint(0, self.num_classes, (batch_size, 1))
      gen_imgs = self.generator.predict([noise, sampled_labels])
      img_labels = y_train[idx]
      d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
      d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
      d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
      g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
      print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))
      if epoch % sample_interval == 0:
        self.sample_images(epoch)
      if epoch > 100 and epoch % 5 == 0:
        self.generate_images(50,epoch)

  def sample_images(self, epoch):
    noise = np.random.normal(0, 1, (self.num_classes, self.latent_dim))
    sampled_labels = np.array([0,1])
    gen_imgs = self.generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5
    img_array = image.img_to_array(gen_imgs[0])
    save_img("Generated3/%dcovid.png" % epoch, img_array)
    img_array = image.img_to_array(gen_imgs[1])
    save_img("Generated3/%dnon-covid.png" % epoch, img_array)

  def generate_images(self, num, epoch=10):
    for i in range(num):
      noise = np.random.normal(0, 1, (1, self.latent_dim))
      sampled_labels = np.array([0])
      gen_imgs = self.generator.predict([noise, sampled_labels])
      # Rescale images 0 - 1
      gen_imgs = 0.5 * gen_imgs + 0.5
      img_array = image.img_to_array(gen_imgs[0])
      save_img("Generated4/%dnon-covid%d.png" % (epoch, i), img_array)
      noise = np.random.normal(0, 1, (1, self.latent_dim))
      sampled_labels = np.array([1])
      gen_imgs = self.generator.predict([noise, sampled_labels])
      # Rescale images 0 - 1
      gen_imgs = 0.5 * gen_imgs + 0.5
      img_array = image.img_to_array(gen_imgs[0])
      save_img("Generated4/%dcovid%d.png" % (epoch, i), img_array)


acgan = ACGAN()
acgan.train(epochs=400, batch_size=16, sample_interval=28)
# acgan.generate_images(100)

