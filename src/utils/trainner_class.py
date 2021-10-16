from vision.probav_isr.data_load import DataLoader
from vision.probav_isr.train_utils import *

import random
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam

import time
import random
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt


class Trainner:
    """
    Clase para el entrenamiento de las redes con el instance noise
    con decay, diferentes pesos entre discriminador y generador.
    Para futuras funcionalidades en los entrenamientos introducirlas aquí.
    """

    def __init__(self, gan, dataloader, base_path,
                 decay_coef=1, blocks=32, 
                 dis_lr=1e-2, gen_lr=1e-4,
                 decay=1e-8, dis_weight=1e-3,
                 gen_weight=5*1e-3):

        self.analytics = {"gloss": [],
                          "dloss": [],
                          }

        blocks = blocks
        lr_dis = dis_lr
        lr_gen = gen_lr
        decay = decay
        dis_weight = dis_weight
        gen_weight = gen_weight

        self.Srgan = gan
        self.data_loader = dataloader
        self.data_len = len(self.data_loader.all_scenes_paths(base_path))

        discr_optimizer = Adam(lr=lr_dis, clipvalue=1.0, decay=decay)
        gan_optimizer = Adam(lr=lr_gen, clipvalue=1.0, decay=decay)

        self.generator = self.Srgan.generator()

        self.discriminator = self.Srgan.discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=discr_optimizer)

        self.srgan = self.Srgan.srgan(self.generator, self.discriminator)
        self.srgan.compile(loss=[cMSE, 'binary_crossentropy'],
                           loss_weights=[gen_weight, dis_weight],
                           optimizer=gan_optimizer)
        self.decay_coef = decay_coef

    def train(self, epochs=2000, batch_size=10, intervalo=200, ronda=0):

        start_time = datetime.datetime.now()
        imgs_hr, imgs_lr = self.data_loader.load_data(self.data_len)
        batch_imgs_index = self.get_batches(
            self.data_len,
            batch_size
        )
        generator_loss = []
        dscriminator_loss = []
        print(epochs)
        for epoch in range(epochs):

            d_losses = []
            g_losses = []

            print('-' * 10, 'Epoca %d' % (epoch + 1), '-' * 10)
            for batch_idx in batch_imgs_index:

                batch = imgs_lr[batch_idx]
                batch_lbl = imgs_hr[batch_idx]
                batch_len = len(batch)
                # Discriminador
                fake_hr = self.generator.predict(batch)
                valida = np.ones(batch_len) - \
                         np.random.random(batch_len) * \
                         ((1 * self.decay_coef) / (epoch + 1))
                fake = np.random.random(batch_len) * \
                       ((1 * self.decay_coef) / (epoch + 1))

                self.discriminator.trainable = True

                d_loss_real = self.discriminator.train_on_batch(batch_lbl, valida)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)
                print("reales",d_loss_real)
                print("falsas",d_loss_fake)
                d_losses.append(0.5 * np.add(d_loss_real, d_loss_fake))
                print("end discriminator")
                # Generador

                select = np.random.choice(range(len(batch_imgs_index)),1)
                imgs_lr_aux = imgs_lr[select]
                imgs_hr_aux = imgs_hr[select]

                fake_hr_aux = self.generator.predict(imgs_lr_aux)
                valida = np.zeros(len(imgs_lr_aux))

                self.discriminator.trainable = False

                g_loss = self.srgan.train_on_batch(
                    imgs_lr_aux,
                    [imgs_hr_aux, valida]
                )
                #  GAN-GENERATOR-DISCRIMINATOR
                g_losses.append(g_loss[1])
                print("end generator")
                elapsed_time = datetime.datetime.now() - start_time

                #print("iteration {}/{} en época {}/{}".format(iteration + 1,
                #                                              data_len // batch_size,
                #                                              epoch + 1,
                #                                              epochs))
                print("         time:           %s" % elapsed_time)

                print("plots") 
                self.plot_multiple_axis(g_losses,d_losses)

            self.generator.save('../resultados/gen_model{}.h5'.format(epoch + 1))
            self.discriminator.save('../resultados/dis_model{}.h5'.format(epoch + 1))
            self.srgan.save('../resultados/gan_model{}.h5'.format( epoch + 1))

            self.analytics["gloss"].append(np.mean(g_losses))
            self.analytics["dloss"].append(np.mean(d_losses))
            self.save_process(self.analytics, epoch)
            print('discriminator loss', np.mean(d_losses))
            print('generator loss', np.mean(g_losses))

    @staticmethod
    def get_batches(length, batch_size):
        values = np.arange(length)
        np.random.shuffle(values)
        comb = np.array_split(values, np.round(length/batch_size))

        return comb

    @staticmethod
    def plot_multiple_axis(v1, v2):

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        ax1.set_xlabel('time (s)')
        ax1.set_ylabel('exp', color=color)
        ax1.plot(v1, color=color)
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  
        color = 'tab:blue'
        ax2.set_ylabel('sin', color=color)  
        ax2.plot(v2, color=color)
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()  
        plt.show()

    @staticmethod
    def save_process(statics, epoch):
        pd.to_pickle(statics, f"../results/statics{epoch}")


if __name__ == "__main__":
    from srgan_model import SRGAN

    tr = Trainner(SRGAN(blocks=4), DataLoader(), "../data/train/")
    tr.data_loader.dataset_name = "../data/"
    tr.train(batch_size=100, epochs=5)
    # model = tf.keras.models.load_model(
    #     '/workspace/resultados/gan_model1.h5',
    #     compile=False
    #     )
    # model.compile(loss=[cMSE, 'binary_crossentropy'],
    #                        loss_weights=[0.8, 0.2],
    #                        optimizer="adam")
    # a,b=tr.data_loader.load_data(10)
    # pred = model.predict(b)
