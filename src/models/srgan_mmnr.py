import os
import numpy as np
import skimage.io
import matplotlib.pyplot as plt

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.activations import LeakyReLU, PReLU
from tensorflow.keras.layers import add
from tf.keras.optimizers import Adam

from vision.probav_isr.srgan_base import SrganBase


class SrganMmnr(SrganBase):

    """
    Estructura del modelo base SRGAN Multimage modified non registered
    para el desarrollo de la investigación presentada en este repositorio.
    """

    def __init__(self, channels=1, lrdim=128, hrdim=384,
                 blocks=16, number_branches=9, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.lr_dim = lrdim
        self.lr_shape = (self.lr_dim, self.lr_dim, self.channels)
        self.hr_dim = hrdim
        self.hr_shape = (self.hr_dim, self.hr_dim, self.channels)
        self.res_block = blocks
        self.number_branches = number_branches

    def generator_branch(self, entrada):
        """
        Construcción del modelo generador mediante el uso de bloques.
        """

        model = Conv2D(filters=64, kernel_size=9,
                       strides=1, padding="same")(entrada)
        model = PReLU(shared_axes=[1, 2])(model)

        gen_model = model

        # Uso n Residual Blocks
        for _ in range(self.res_block):
            model = self.res_block_gen(model, 3, 64, 1)

        model = Conv2D(filters=64, kernel_size=3, strides=1,
                       padding="same")(model)
        model = BatchNormalization(momentum=0.3)(model)
        model = add([gen_model, model])

        # Uso 1 UpSampling Block
        model = self.up_sampling_block(model, 3, 256, 1)

        model = Conv2D(filters=self.channels, kernel_size=9,
                       strides=1, padding="same")(model)
        model = Activation('sigmoid')(model)

        return model

    def generator(self):

        branches = []
        gen_inputs = []
        for num, branch in enumerate(range(self.number_branches)):
            gen_inputs.append(Input(shape=self.lr_shape))
            branches.append(self.generator_branch(gen_inputs[num]))
        model = keras.layers.average(branches)

        generator_model = Model(inputs=gen_inputs, outputs=model)

        return generator_model

    def discriminator(self):
        """
        Desarrollo del modelo discriminador mediante el uso de bloques.
        """

        dis_input = Input(shape=self.hr_shape)

        model = Conv2D(filters=32, kernel_size=3, strides=1,
                       padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = self.discriminator_block(model, 32, 3, 2)
        for i in [1, 2, 3]:
            model = self.discriminator_block(model, 64 * i, 3, 1)
            model = self.discriminator_block(model, 64 * i, 3, 2)

        model = Flatten()(model)
        model = Dense(512)(model)
        model = LeakyReLU(alpha=0.1)(model)

        model = Dense(1)(model)
        model = Activation('sigmoid')(model)

        discriminator_model = Model(inputs=dis_input, outputs=model)

        return discriminator_model

    def srgan(self, generator, discriminator):
        """
        Construcción de la SRGAN.
        """

        discriminator.trainable = False
        srgan_in = []
        for num in range(self.number_branches):
            srgan_in.append(Input(shape=self.lr_shape))
        gen_out = generator(srgan_in)
        srgan_out = discriminator(gen_out)

        return Model(srgan_in, [gen_out, srgan_out])

    @staticmethod
    def res_block_gen(model, kernal_size, filters, strides):

        gen = model

        model = Conv2D(filters=filters, kernel_size=kernal_size,
                       strides=strides, padding="same")(model)
        model = BatchNormalization()(model)
        model = PReLU(shared_axes=[1, 2])(model)
        model = Conv2D(filters=filters, kernel_size=kernal_size,
                       strides=strides, padding="same")(model)
        model = BatchNormalization()(model)

        model = add([gen, model])

        return model

    @staticmethod
    def up_sampling_block(model, kernal_size, filters, strides):
        model = Conv2D(filters=filters, kernel_size=kernal_size,
                       strides=strides, padding="same")(model)
        model = LeakyReLU(alpha=0.2)(model)
        model = UpSampling2D(size=3)(model)
        model = Conv2D(filters=filters, kernel_size=kernal_size,
                       strides=strides, padding="same")(model)
        model = LeakyReLU(alpha=0.25)(model)

        return model

    @staticmethod
    def discriminator_block(model, filters, kernel_size, strides):

        model = Conv2D(filters=filters, kernel_size=kernel_size,
                       strides=strides, padding="same")(model)
        model = BatchNormalization()(model)
        model = LeakyReLU(alpha=0.1)(model)

        return model
