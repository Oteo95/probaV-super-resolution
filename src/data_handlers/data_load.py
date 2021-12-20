import numpy as np
import pandas as pd
from glob import glob
from tqdm import tqdm
import skimage
import skimage.io
import os
import itertools
import scipy


class DataLoader:
    """
    Clase para la carga y transformación de los datos. En esta
    clase se han de introducir todos los elementos necesarios para
    la carga de los diferentes canales o imágenes así como sus
    transformaciones.
    """

    def __init__(self, data_path="../data/"):
        self.dataset_name = data_path

    @staticmethod
    def all_scenes_paths(base_path):
        """
        Obtención de los paths de las imágenes.
        """
        if base_path[-1] not in {'/'}:
            base_path = base_path + '/'
        return [base_path + c + s
                for c in ['RED/', 'NIR/']
                for s in sorted(os.listdir(base_path + c))]

    @staticmethod
    def lowres_image_iterator(path, img_as_float=True):
        """
        Iterador para la obtención de las imágenes LR.
        Cada llamada al iterador devuelve una imagen LR.
        Inputs:
        path: Path de la imagen.
        img_as_float: Mantener siempre como True.
        """

        if path[-1] in {'/'}:
            path = path
        else:
            path = path + '/'

        for f in glob(path + 'LR*.png'):
            q = f.replace('LR', 'QM')
            lr = skimage.io.imread(f, as_gray=True)
            cm = skimage.io.imread(q, as_gray=True)
            if img_as_float:
                lr = skimage.img_as_float(lr)
            yield (lr, cm)

    @staticmethod
    def highres_image(path, img_as_float=True):
        """
        Función para la obtención de las HR.
        Inputs:

        path: Path de la imagen HR.
        img_as_float: Mantener siempre como True
        """

        if path[-1] in {'/', '\\'}:
            path = path
        else:
            path = path + '/'

        hr = skimage.io.imread(path + 'HR.png', as_gray=True)
        sm = skimage.io.imread(path + 'SM.png', as_gray= True)
        if img_as_float:
            hr = skimage.img_as_float(hr)
        return hr, sm

    def transformacion_inputs(self, images, agg_with='median',
                              only_clear=False, fill_obscured=False,
                              img_as_float=True):
        """
        Tranformación de los datos por la media o la mediana.
        Inputs:

        imagenes: Imágenes de baja resolución.
        agg_width: Método de agregación de las imágenes.
        Others: Mantener el resto de argumentos por defecto.
        """
        agg_opts = {
            'mean': lambda i: np.nanmean(i, axis=0),
            'median': lambda i: np.nanmedian(i, axis=0),
            }
        agg = agg_opts[agg_with]

        imgs = []
        obsc = []

        if isinstance(images, str):
            images = self.lowres_image_iterator(images,
                                                img_as_float or only_clear)
        elif only_clear:
            images = [(lr.copy(), cm) for (lr, cm) in images]

        for (lr, cm) in images:

            if only_clear:
                if fill_obscured is not False:
                    o = lr.copy()
                    o[cm] = np.nan
                    obsc.append(o)
                lr[~cm] = np.nan
            imgs.append(lr)

        with np.warnings.catch_warnings():
            np.warnings.filterwarnings('ignore',
                                       r'All-NaN (slice|axis) encountered')
            np.warnings.filterwarnings('ignore', r'Mean of empty slice')

            agg_img = agg(imgs)

            if only_clear and fill_obscured is not False:
                if isinstance(fill_obscured, str):
                    agg = agg_opts[fill_obscured]
                some_clear = np.isnan(obsc).any(axis=0)
                obsc = agg(obsc)
                obsc[some_clear] = 0.0
                np.nan_to_num(agg_img, copy=False)
                agg_img += obsc
        return agg_img

    def load_data(self, batch_size=1):
        """
        Todo el dataset tarda alrededor de 30 segundos
        """
        train = self.all_scenes_paths(self.dataset_name + 'train')
        dataset_size = len(train)
        batch_images = np.random.choice(
            train,
            size=min(batch_size, dataset_size)
        )
        imgs_hr = []
        imgs_lr = []

        for img_path in batch_images:
            hr, sm = self.highres_image(img_path)
            hr[~sm] = 0.01
            img_hr = hr

            img_lr = self.transformacion_inputs(img_path,
                                                agg_with='median',
                                                only_clear=False)

            img_hr = np.expand_dims(img_hr, axis=2)
            img_lr = np.expand_dims(img_lr, axis=2)

            imgs_hr.append(img_hr)
            imgs_lr.append(img_lr)
        return np.array(imgs_hr), np.array(imgs_lr)
