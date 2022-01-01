from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K


class BatchNorm(keras.layers.BatchNormalization):
    '''
    Batch normalization has a negative effect on training if batches are small
    so it is disable here.
    '''
    def call(self, inputs, training: bool = None):
        return super().call(inputs, training=False)

class ConvModelBuilder:
    def __init__(self, input_shape: tuple):
        self.input_shape = input_shape

        # TODO: delete this since not using config file to
        # build the model.
        self.hash_map = {
            "conv2d": layers.Conv2D,
            "batchnorm": BatchNorm,
            "conv2dT": layers.Conv2DTranspose,
            "activation": layers.Activation,
            "maxpooling": layers.MaxPooling2D,
            "upsampling2d": layers.UpSampling2D,
            "bblock": self.begin_block,
            "resblock": self.residual_block,
            "center": self.center_block,
            "decoder": self.decoder_block,
            "encoder": self.encoder_block,
            "upblock": self.upsampling_block,
        }

    def begin_block(
        self,
        input_tensor,
        activation: str,
        layers: int=2
    ):

        x = self.hash_map["conv2d"](
            64,
            (3, 3),
            padding='same',
            activation=activation,
            kernel_initializer='he_normal'
        )(input_tensor)
        x = BatchNorm()(x)

        for _ in range(layers-1):
            x = self.hash_map["conv2d"](
                64,
                (3, 3),
                padding='same',
                activation=activation,
                kernel_initializer='he_normal'
                )(x)
            x = BatchNorm()(x)
        skip = x
        return skip, x

    def residual_block(
        self,
        input_tensor,
        num_filters: int,
        activation: str = 'relu',
        end_activation: bool = True
    ):
        x = layers.Conv2D(
            num_filters,
            (3, 3),
            padding='same',
            activation=activation,
            kernel_initializer='he_normal'
        )(input_tensor)
        x = self.hash_map["batchnorm"]()(x)
        x = layers.Conv2D(
            num_filters,
            (3, 3),
            padding='same',
            activation='linear',
            kernel_initializer='he_normal'
        )(x)
        x = self.hash_map["batchnorm"]()(x)

        shortcut_shape = input_tensor.shape
        residual_shape = x.shape
        if shortcut_shape!=residual_shape:
            num_filters = residual_shape[-1]
            input_tensor = layers.Conv2D(
                num_filters,
                (1, 1),
                padding='same',
                activation=activation,
                kernel_initializer='he_normal'
            )(input_tensor)
        x= layers.Add()([input_tensor, x])

        if end_activation:
            x = layers.Activation(activation)(x)
        return x

    def encoder_block(
        self,
        input_tensor,
        num_filters: int ,
        activation: str = 'relu',
        downsample_type: str = 'maxpooling'
    ):
        encoder = self.hash_map["resblock"](input_tensor, num_filters)
        if downsample_type=='maxpooling':
            encoder_pooled = self.hash_map["maxpooling"](
                pool_size=(2, 2)
            )(encoder)
        elif downsample_type=='conv2d':
            encoder_pooled = self.hash_map["conv2d"](
                num_filters,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding='valid',
                activation=activation,
                kernel_initializer='he_normal'
            )(encoder)
        else:
            raise ValueError(
                'Unknown downsample_type:',
                downsample_type
            )
        return encoder, encoder_pooled

    def center_block(self, input_tensor, num_filters, activation='relu'):
        block = self.hash_map["resblock"](
            input_tensor,
            num_filters,
            activation=activation
        )
        return block

    def decoder_block(
        self,
        input_tensor,
        skip_tensor,
        num_filters,
        activation='relu',
        upsample_type='conv2dT',
        skip_type='concatenate',
        end_activation=True
    ):
        if upsample_type=='upsampling2d':
            decoder = self.hash_map["upsampling2d"](
                size=(2, 2)
            )(input_tensor)
        elif upsample_type=='conv2dT':
            decoder = self.hash_map["conv2dT"](
                num_filters,
                kernel_size=(2, 2),
                strides=(2, 2),
                padding='same',
                activation=activation,
                kernel_initializer='he_normal'
            )(input_tensor)
        else:
            raise ValueError('Unknown upsample_type:', upsample_type)

        if skip_type=='concatenate':
            decoder = layers.Concatenate(
                axis=-1
            )([decoder, skip_tensor])

        elif skip_type=='add':
            decoder = layers.Add()([decoder, skip_tensor])
        else:
            raise ValueError('Unknown skip_type:', skip_type)
            
        decoder = self.hash_map["resblock"](
            decoder,
            num_filters,
            activation=activation,
            end_activation=end_activation)
        return decoder

    def upsampling_block(self, input_tensor, activation, nlayers: int = 2):
        x = layers.Conv2DTranspose(
            64,
            kernel_size=(3, 3),
            strides=(3, 3),
            padding='same',
            activation=activation,
            kernel_initializer='he_normal'
        )(input_tensor)
        x = BatchNorm()(x)
        for _ in range(nlayers):
            x = layers.Conv2D(
                64,
                (3, 3),
                padding='same',
                activation=activation,
                kernel_initializer='he_normal'
            )(x)
            x = BatchNorm()(x)
        
        x = layers.Conv2D(
                1,
                (1, 1),
                padding='same',
                activation='linear',
                kernel_initializer='he_normal'
            )(x)
        
        return x

    def umodel(self, activation: str = 'relu', downsample_type: str = 'maxpooling'):
        input_layer = layers.Input(shape=self.input_shape)
        skip, x = self.hash_map["bblock"](
            input_layer,
            activation=activation,
            layers=2
        )
        encoder0, encoder_pool0 = self.hash_map["encoder"](
            x,
            64,
            activation=activation
        )
        encoder, encoder_pool = self.hash_map["encoder"](
            encoder_pool0,
            64,
            activation=activation,
            downsample_type = downsample_type
        )
        encoder2, encoder_pool2 = self.hash_map["encoder"](
            encoder_pool,
            128,
            activation=activation,
            downsample_type = downsample_type
        )
        encoder3, encoder_pool3 = self.hash_map["encoder"](
            encoder_pool2,
            256,
            activation=activation,
            downsample_type = downsample_type
        )

        center = self.hash_map["center"](
            encoder_pool3,
            512,
            activation=activation
        )

        decoder = self.hash_map["decoder"](
            center,
            encoder3, 256,
            activation=activation
        )
        decoder2 = self.hash_map["decoder"](
            decoder,
            encoder2,
            128,
            activation=activation
        )
        decoder3 = self.hash_map["decoder"](
            decoder2,
            encoder,
            64,
            activation=activation
        )
        decoder4 = self.hash_map["decoder"](
            decoder3,
            encoder0,
            64,
            activation=activation
        )

        x = self._addition_layers(
            decoder4,
            skip,
            activation=activation,
            skip_method="add"
        )
        output = self.hash_map["upblock"](x, activation, nlayers=2)
        
        model = Model(inputs=input_layer, outputs=output)
        
        return model

    @staticmethod
    def _addition_layers(enter_layer, skip, activation: str, skip_method: str = "add"):
        if skip_method=='add':
            x = layers.Add()([enter_layer, skip])
            x = layers.Activation(activation)(x)
        elif skip_method=='concatenate':
            x = layers.Activation(activation)(enter_layer)
            x = layers.Concatenate()([x, skip])
        else:
            raise ValueError('Invalid long_skip:', skip_method)

        return x

def PSNR(y_true, y_pred):
    '''Peak Signal to Noise Ratio metric for Keras.'''
    eps = 1e-8
    mpixel = 1.0
    log_ = K.log(
        (mpixel ** 2) / (K.mean(K.square(y_pred - y_true + eps), axis=-1))
    )
    return (10.0 * log_) / 2.303




if __name__ == "__main__":
    cnnbuilder = ConvModelBuilder((128,128,1))
    model = cnnbuilder.umodel(activation="relu")
    from data_handlers.data_load import DataLoader
    data_loader = DataLoader()
    a,b = data_loader.load_data(1000)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss="mse")
    model.fit(
        b,a, epochs=100, validation_split=0.2,
        batch_size=4,
        verbose=True)

    preds = model.predict(b[100:102])
    import matplotlib.pyplot as plt
    plt.imshow(preds[0])