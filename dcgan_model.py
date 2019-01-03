# -------------------- Optimized by https://github.com/soumith/ganhacks -----------------------
#                    If you are interested in detailed optimization, see that URL
# ---------------------------------END DESCRIPTION---------------------------------------------

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, RepeatVector, multiply
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD

import matplotlib.pyplot as plt

import sys

import numpy as np

# Fix Error
class LeakyReLU(LeakyReLU):

    def __init__(self, **kwargs):
        self.__name__ = "LeakyReLU"
        super(LeakyReLU, self).__init__(**kwargs)

def generate_random_arr_between(a, b, shape):
    return a + (b - a) * np.random.random(shape)


class DCGAN():
    def __init__(self):
        # Input shape
        self.time_length = 5000
        self.channels = 7
        self.signal_shape = (self.time_length, self.channels)
        # self.img_rows = 28
        # self.img_cols = 28
        # self.channels = 1
        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0004, 0.5)
        sgd_optimizer = SGD(0.00005, momentum=0.9)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=sgd_optimizer,
                                   metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([z, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator([img, label])

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([z, label], valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = Sequential()

        model.add(Dense(128 * 1250, activation=LeakyReLU(), input_dim=self.latent_dim))
        model.add(Reshape((1250, 128)))

        model.add(UpSampling1D())
        model.add(Conv1D(128, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(UpSampling1D())
        model.add(Conv1D(64, kernel_size=3, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU())
        model.add(Conv1D(self.channels, kernel_size=3, padding="same"))
        # change to LeakyRelu since a lot of signal exceeding 1
        model.add(LeakyReLU())

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        # img = model(noise)

        label = Input(shape=(1,))
        label_embedding = Flatten()(RepeatVector(self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Reshape((self.time_length, self.channels)))

        model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=self.signal_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding1D(padding=0))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Conv1D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        img = Input(shape=self.signal_shape)

        label = Input(shape=(1,))
        # reason is referred as before
        label_embedding = Flatten()(RepeatVector(np.prod(self.signal_shape))(label))

        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        from phm_dataset import PHMToolWearDataset
        tool_wear_dataset = PHMToolWearDataset()
        x, y = tool_wear_dataset.get_all_data
        # print(x.shape)
        y = y.max(axis=1)
        X_train, y_train = x, y

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Adversarial ground truths
            # replace them with random label -> # 7
            if not(epoch % 3 == 0 and epoch % 50 != 0):
                # Soft adjustment
                valid = np.ones((batch_size, 1))
                fake = np.zeros((batch_size, 1))
            else:
                valid = generate_random_arr_between(0, 0.3, (batch_size, 1))
                fake = generate_random_arr_between(0.7, 1.2, (batch_size, 1))


            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator (real classified as ones and generated as zeros)
            # (imgs.shape, gen_imgs.shape)
            d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
            d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels (tool wear 0-300)
            sampled_labels = np.random.randint(0+10, 300+10, batch_size).reshape(-1, 1)

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.save_model()

    def save_imgs(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.arange(0+10, 300+10, 30).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise,sampled_labels])






        for channel in range(7):
            fig, axs = plt.subplots(r, c)
            import os
            directory_path = os.path.join("images","%s"%(channel+1))
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    # Force in Y
                    axs[i, j].plot(gen_imgs[cnt, :, channel])
                    axs[i, j].set_title("%s " % (sampled_labels[cnt][0]))
                    axs[i, j].axis('off')
                    cnt += 1
            fig.savefig("images/%s/channel_%s_epoch_%d.png" % (channel+1,channel+1,epoch))
            plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "soft_dcgan_generator")
        save(self.discriminator, "soft_dcgan_discriminator")


if __name__ == '__main__':
    dcgan = DCGAN()
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
