import tensorflow.keras.backend as K
import tensorflow as tf


def cMSE(hr, sr):
    """
    Función de perdida del problema para la super-resolución de
    imágenes hiper-espectrales NIR y RED.
    """

    obs = tf.equal(hr, 0.05)
    clr = tf.math.logical_not(obs)
    _hr = tf.boolean_mask(hr, clr)
    _sr = tf.boolean_mask(sr, clr)

    # Se cálcula el bias: b
    pixel_diff = _hr - _sr
    b = K.mean(pixel_diff)

    # Se cálcula el clear mean-square error corregido
    pixel_diff = pixel_diff - b
    cMse = K.mean(pixel_diff * pixel_diff)

    return cMse
