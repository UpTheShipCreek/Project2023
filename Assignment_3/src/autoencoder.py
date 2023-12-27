import tensorflow as tf
from tensorflow.keras import layers, models

def build_autoencoder(input_shape, latent_dim):
    model = models.Sequential()

    # Encoder
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.MaxPooling2D((2, 2), padding='same'))
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dim, activation='relu', name='latent_space'))

    # Decoder
    model.add(layers.Dense(7 * 7 * 64, activation='relu'))
    model.add(layers.Reshape((7, 7, 64)))
    model.add(layers.Conv2DTranspose(64, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(32, (3, 3), activation='relu', padding='same'))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'))

    return model

def normalize_to_integer(array):
    return (array * 255).astype('uint8')

def save_model(model, filepath):
    model.save(filepath)

def load_model(filepath):
    return tf.keras.models.load_model(filepath, compile=False)
