# -------------------- Optimized by https://github.com/soumith/ganhacks -----------------------
#                    If you are interested in detailed optimization, see that URL
# ---------------------------------END DESCRIPTION---------------------------------------------

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, RepeatVector, multiply, Embedding
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, ZeroPadding1D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, UpSampling1D, Conv1D
from keras.models import Sequential, Model
from keras.optimizers import Adam, SGD
from keras_contrib.layers import InstanceNormalization
from keras.utils import to_categorical
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

def normalize_signal_between(signal, a:float, b:float, min:float, max:float):
    return a + (b - a) / (max - min) * (signal - min)


CHANNEL_MAX = [178.4, 101.01, 84.633, 2.094, 2.316, 3.353, 0.63]
CHANNEL_MIN = [-7.8787e+01, -4.4339e+01, -4.2628e+01, -2.2670e+00, -3.0220e+00, -3.2540e+00, -5.0000e-03]


def normalize_signal_to_sample(signal:np.ndarray):
    normalized_signal = np.zeros(signal.shape)
    for channel_idx in range(7):
        normalized_signal[:,:,channel_idx] = normalize_signal_between(signal[:,:,channel_idx],0,1,
                                                                      CHANNEL_MIN[channel_idx],
                                                                      CHANNEL_MAX[channel_idx])
    return normalized_signal

def normalize_sample_to_signal(signal:np.ndarray):
    sample_signal = np.zeros(signal.shape)
    for channel_idx in range(7):
        sample_signal[:, :, channel_idx] = normalize_signal_between(signal[:, :, channel_idx],
                                                                        CHANNEL_MIN[channel_idx],
                                                                        CHANNEL_MAX[channel_idx],
                                                                        0, 1,)
    return sample_signal

class DCGAN():
    def __init__(self):
        # Input shape
        self.time_length = 500
        self.channels = 7
        self.signal_shape = (self.time_length, self.channels)
        # self.img_rows = 28
        # self.img_cols = 28
        # self.channels = 1
        # self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 3500
        self.num_classes = 50

        optimizer = Adam(0.0001, 0.5)
        sgd_optimizer = SGD(0.0001)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
        self.MAX_TOOL_WEAR = 300

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

        model.add(Dense(128 * 125, activation=LeakyReLU(), input_dim=self.latent_dim))
        model.add(Reshape((125, 128)))

        model.add(UpSampling1D())
        model.add(Conv1D(128, kernel_size=3, padding="same"))
        model.add(LeakyReLU())
        model.add(InstanceNormalization())
        # model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling1D())
        model.add(Conv1D(128, kernel_size=3, padding="same"))
        model.add(LeakyReLU())
        model.add(InstanceNormalization())

        model.add(UpSampling1D())
        model.add(Conv1D(256, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU())
        model.add(InstanceNormalization())
        # model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling1D())
        model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU())
        model.add(InstanceNormalization())

        model.add(Conv1D(self.channels, kernel_size=3, padding="same"))
        # change to LeakyRelu since a lot of signal exceeding 1
        model.add(Activation("tanh"))

        model.summary()

        noise = Input(shape=(self.latent_dim,))
        # img = model(noise)

        label = Input(shape=(1,))
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.signal_shape))(label))
        # label_embedding = Flatten()(RepeatVector(self.latent_dim)(label))
        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

    def build_discriminator(self):

        model = Sequential()
        model.add(Reshape((self.time_length, self.channels)))

        model.add(Conv1D(32, kernel_size=3, strides=2, input_shape=self.signal_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Conv1D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding1D(padding=0))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(InstanceNormalization())
        # model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(InstanceNormalization())
        # model.add(BatchNormalization(momentum=0.8))

        model.add(Conv1D(256, kernel_size=3, strides=1, padding="same"))
        # model.add(InstanceNormalization())
        # model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        # model.summary()

        img = Input(shape=self.signal_shape)

        label = Input(shape=(1,))
        # reason is referred as before
        label_embedding = Flatten()(Embedding(self.num_classes, np.prod(self.signal_shape))(label))
        # label_embedding = Flatten()(RepeatVector(np.prod(self.signal_shape))(label))

        flat_img = Flatten()(img)


        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        from phm_dataset import PHMToolWearDataset
        tool_wear_dataset = PHMToolWearDataset()
        # x, y = tool_wear_dataset.get_recoginition_data_in_class_num(class_num=50)
        x, y = tool_wear_dataset.get_reinforce_recoginition_data_in_class_num(class_num=50)
        # x = x[:, ::10, :]
        (X_train, y_train) = x, y

        X_train = normalize_signal_to_sample(X_train)

        # # Rescale -1 to 1
        # X_train = X_train / 127.5 - 1.
        # X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # Adversarial ground truths
            # replace them with random label -> # 7
            if not (epoch % 3 == 0) or True:
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
            # labels = to_categorical(labels, num_classes=self.num_classes)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict([noise, labels])

            # Train the discriminator (real classified as ones and generated as zeros)
            # (imgs.shape, gen_imgs.shape)

            # make the labels noisy for the discriminator
            if epoch % 79 == 0 and False:
                # flip the right to wrong
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], fake)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], valid)
            else:
                d_loss_real = self.discriminator.train_on_batch([imgs, labels], valid)
                d_loss_fake = self.discriminator.train_on_batch([gen_imgs, labels], fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Condition on labels (tool wear 0-300)
            sampled_labels = np.random.randint(0, 50, batch_size).reshape(-1, 1)

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch([noise, sampled_labels], valid)

            # Plot the progress
            # print(self.discriminator.metrics_names)
            # print(d_loss_real,"\n",d_loss_fake,"\n",d_loss)

            print("%d [D loss: %f, acc.: %.2f%%(True %.2f%%, Fake %.2f%%)] [G loss: %f]" % (
            epoch, d_loss[0], 100 * d_loss[1], 100 * d_loss_real[1], 100 * d_loss_fake[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)

                self.save_model(epoch=epoch)

            if epoch % 1000 == 0:
                self.generate_and_test_signal(filename="images/Sample_at_%s_epoch.svg" % (epoch))

    def save_imgs(self, epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        sampled_labels = np.arange(0, self.num_classes).reshape(-1, 1)
        gen_imgs = self.generator.predict([noise, sampled_labels])
        gen_imgs = normalize_sample_to_signal(gen_imgs)

        for channel in range(7):
            fig, axs = plt.subplots(r, c)
            import os
            directory_path = os.path.join("images", "%s" % (channel + 1))
            if not os.path.exists(directory_path):
                os.mkdir(directory_path)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    # Force in Y
                    axs[i, j].plot(gen_imgs[cnt, :, channel], label="%d" % (cnt))
                    axs[i, j].set_title("%s " % (sampled_labels[cnt][0]))
                    # axs[i, j].legend()
                    # axs[i, j].axis('off')

                    cnt += 1
            fig.savefig("images/%s/channel_%s_epoch_%d.png" % (channel + 1, channel + 1, epoch))
            plt.close("all")

    def save_model(self,epoch=None):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                       "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "dcgan_generator_DS_REIN_SCALE_01")
        save(self.discriminator, "dcgan_discriminator_DS_REIN_SCALE_01")
        if epoch != None:
            save(self.generator, "dcgan_generator_DS_REIN_SCALE_01@%s"%(epoch))
            save(self.discriminator, "dcgan_discriminator_DS_REIN_SCALE_01@%s"%(epoch))

    def load_model(self,epoch=None):

        def load(model, model_name):
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            model.load_weights(weights_path)

        load(self.generator, "dcgan_generator_DS_REIN_SCALE_01")
        load(self.discriminator, "dcgan_discriminator_DS_REIN_SCALE_01")
        if epoch:
            load(self.generator, "dcgan_generator_DS_REIN_SCALE_01@%s"%(epoch))
            load(self.discriminator, "dcgan_discriminator_DS_REIN_SCALE_01@%s"%(epoch))

    def generate_and_test_signal(self, filename=None):
        batch_number = self.num_classes
        sampled_noise = np.random.normal(0, 1, (batch_number, self.latent_dim))
        # noise = np.random.normal(0, 1, (batch_number, self.latent_dim))
        sampled_labels = np.arange(0, batch_number).reshape(-1, 1)
        labels = np.arange(0, batch_number).reshape(-1, 1)
        # label = sampled_labels
        # sampled_labels = to_categorical(sampled_labels, num_classes=self.num_classes)

        # gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)
        gen_imgs = self.generator.predict([sampled_noise,sampled_labels])
        gen_imgs = normalize_sample_to_signal(gen_imgs)
        # gen_imgs = self.generator.predict([noise, sampled_labels])

        from resnet_model import build_multi_input_main_residual_network
        model = build_multi_input_main_residual_network(32, 500, 7, 1, loop_depth=20)
        model.load_weights("Resnet_block_20_downSample_toolMax.kerascheckpts")

        y_pred = model.predict(gen_imgs)
        # print(y_pred.max(axis=1).shape,y_pred.shape)
        import matplotlib.pyplot as plt
        fig = plt.figure()
        plt.plot(labels * (210 / 50) + 30, label="Aimed Tool wear")
        plt.plot(y_pred, label="Predicted by ResNet")
        plt.ylabel("Tool Wear $(\mu m)$")
        plt.xlabel("Sample")
        plt.legend()
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
        plt.close("all")


if __name__ == '__main__':
    dcgan = DCGAN()
    PREDICT = False

    if PREDICT:
        EPOCH = 258000
        dcgan.load_model(epoch=EPOCH)

        dcgan.generate_and_test_signal()
        for i in range(20):
            dcgan.generate_and_test_signal(filename="GAN_TEST_%s_EPOCH_%s.svg" %(i,EPOCH))
    else:
        dcgan.train(epochs=40000+1, batch_size=16, save_interval=2000)
