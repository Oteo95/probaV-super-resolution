from data_handlers.data_load import DataLoader
from utils.train_utils import *
import os

from tensorflow.keras.optimizers import Adam

import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

import tensorflow as tf


class Trainner:
    """
    Clase para el entrenamiento de las redes con el instance noise
    con decay, diferentes pesos entre discriminador y generador.
    Para futuras funcionalidades en los entrenamientos introducirlas aquí.
    """

    physical_devices = tf.config.list_physical_devices('GPU') 
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

    def __init__(self, gan, dataloader, base_path,
                 decay_coef=1, blocks=32, 
                 dis_lr=1e-6, gen_lr=1e-4,
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

    def train(self, epochs=2000, batch_size=10, intervalo=1, ronda=0):

        path_orig = "/workspaces/probaV-super-resolution/results/"
        time_path = str(datetime.datetime.now())
        os.mkdir(f"{path_orig}{time_path}")

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
            epoch_start = datetime.datetime.now()
            for num, batch_idx in enumerate(batch_imgs_index):
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

                if epoch%30==0:
                    self.discriminator.trainable = True
                else:
                    self.discriminator.trainable = False

                d_loss_real = self.discriminator.train_on_batch(batch_lbl, valida)
                d_loss_fake = self.discriminator.train_on_batch(fake_hr, fake)

                d_losses.append(0.5 * np.add(d_loss_real, d_loss_fake))
                # Generador

                select = np.random.choice(range(len(batch_imgs_index)),1)
                imgs_lr_aux = imgs_lr[select]
                imgs_hr_aux = imgs_hr[select]

                # fake_hr_aux = self.generator.predict(imgs_lr_aux)
                valida = np.zeros(len(imgs_lr_aux))

                self.discriminator.trainable = False

                g_loss = self.srgan.train_on_batch(
                    imgs_lr_aux,
                    [imgs_hr_aux, valida]
                )
                
                #  GAN-GENERATOR-DISCRIMINATOR
                g_losses.append(g_loss[1])
                #print("iteration {}/{} en época {}/{}".format(iteration + 1,
                #                                              data_len // batch_size,
                #                                              epoch + 1,
                #                                              epochs))
                
                #print("         time:           %s" % elapsed_time)

                # print("plots") 
                # self.plot_multiple_axis(g_losses,d_losses)
                if num%100==0:
                    elapsed_time2 = datetime.datetime.now() - epoch_start
                    print("time 100 batchs: ",elapsed_time2)
                    epoch_start = datetime.datetime.now()
            
            if epoch%10==0:
                self.generator.save(
                    f'{path_orig}{time_path}/gen_model{epoch + 1}.h5'
                )
                self.discriminator.save(
                    f'{path_orig}{time_path}/dis_model{epoch + 1}.h5'
                )
                self.srgan.save(
                    f'{path_orig}{time_path}/gan_model{epoch+1}.h5'
                )

            self.analytics["gloss"].append(np.mean(g_losses))
            self.analytics["dloss"].append(np.mean(d_losses))
            # self.save_process(self.analytics, epoch)
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
        pd.to_pickle(statics, f"/workspaces/probaV-super-resolution/results/statics{epoch}")


if __name__ == "__main__":
    from models.srgan_model import SRGAN

    tr = Trainner(
        SRGAN(blocks=2),
        DataLoader(),
        "/workspaces/probaV-super-resolution/data/train/",

        )
    tr.data_loader.dataset_name = "/workspaces/probaV-super-resolution/data/"
    tr.train(batch_size=16, epochs=200)
    model = tf.keras.models.load_model(
        '/workspaces/probaV-super-resolution/results/2021-12-20 17:30:48.061651/gan_model1.h5',
        compile=False
    )
    model.compile(
        loss=[cMSE, 'binary_crossentropy'],
        loss_weights=[0.8, 0.2],
        optimizer=Adam(learning_rate=0.00005),
    )
    a,b=tr.data_loader.load_data(10)
    pred = model.predict(b)
    for i in pred[0]:
        plt.imshow(i)
        plt.show()
