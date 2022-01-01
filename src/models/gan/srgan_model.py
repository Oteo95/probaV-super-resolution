from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LeakyReLU, PReLU
from tensorflow.keras.layers import add

from models.gan.srgan_base import SrganBase


class SRGAN(SrganBase):
    """
    Estructura del modelo base SRGAN modificado para el desarrollo de 
    la investigación presentada en este repositorio.
    """

    def __init__(self, channels=1, lrdim=128, hrdim=384, blocks=16, **kwargs):
        super().__init__(**kwargs)
        self.channels = channels
        self.lr_dim = lrdim
        self.lr_shape = (self.lr_dim, self.lr_dim, self.channels)
        self.hr_dim = hrdim
        self.hr_shape = (self.hr_dim, self.hr_dim, self.channels)
        self.res_block = blocks

    def generator(self):
        """
        Construcción del modelo generador mediante el uso de bloques.
        """

        gen_input = Input(shape=self.lr_shape)
        model = Conv2D(filters=8, kernel_size=9,
                       strides=1, padding="same")(gen_input)
        model = PReLU(shared_axes=[1, 2])(model)

        gen_model = model

        # Uso n Residual Blocks
        for _ in range(self.res_block):
            model = self.res_block_gen(model, 3, 8, 1)# 64

        model = Conv2D(filters=8, kernel_size=3, strides=1,
                       padding="same")(model)
        model = BatchNormalization(momentum=0.3)(model)
        model = add([gen_model, model])

        # Uso 1 UpSampling Block
        model = self.up_sampling_block(model, 3, 128, 1) #256

        model = Conv2D(filters=self.channels, kernel_size=9,
                       strides=1, padding="same")(model)
        model = Activation('linear')(model)

        generator_model = Model(inputs=gen_input, outputs=model)

        return generator_model

    def discriminator(self):
        """
        Desarrollo del modelo discriminador mediante el uso de bloques.
        """

        dis_input = Input(shape=self.hr_shape)

        model = Conv2D(filters=4, kernel_size=3, strides=1,
                       padding="same")(dis_input)
        model = LeakyReLU(alpha=0.2)(model)

        model = self.discriminator_block(model, 4, 3, 2)
        for i in [1, 2, 3]:
            model = self.discriminator_block(model, 4 * i, 3, 1)
            model = self.discriminator_block(model, 4 * i, 3, 2)

        model = Flatten()(model)
        model = Dense(32)(model)
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
        srgan_in = Input(shape=self.lr_shape)
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
