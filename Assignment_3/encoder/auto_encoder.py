import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import layers, Model, losses
from keras.datasets import mnist


# class Autoencoder(Model):
#     def __init__(self, encoding_dimension):
#         super(Autoencoder, self).__init__()

#         self.encoding_dimension = encoding_dimension

#         # Encoder
#         self.encoder = tf.keras.Sequential([
#             layers.Reshape((28, 28, 1)),
#             layers.Conv2D(78, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-4)),
#             layers.MaxPooling2D((2, 2), padding='same'),
#             layers.Conv2D(encoding_dimension, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-4)),
#             layers.Flatten(),
#             layers.Dense(encoding_dimension),
#         ])

#         # Decoder
#         self.decoder = tf.keras.Sequential([
#             layers.Dense(28 * 28 * encoding_dimension, activation='relu', input_shape=(encoding_dimension,)),
#             layers.Reshape((28, 28, encoding_dimension)),
#             layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'),
#             layers.Reshape((784,))
#         ])

#     def call(self, x):
#         encoded = self.encoder(x)
#         decoded = self.decoder(encoded)
#         return decoded

class Autoencoder(Model):
    def __init__(self, encoding_dimension):
        super(Autoencoder, self).__init__()

        self.encoding_dimension = encoding_dimension

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(78, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-4)),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(encoding_dimension, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-4)),
            layers.Flatten(),
            layers.Dense(encoding_dimension),
        ])

        # Decoder
        self.decoder = tf.keras.Sequential([
            layers.Dense(28 * 28 * encoding_dimension, activation='relu', input_shape=(encoding_dimension,)),
            layers.Reshape((28, 28, encoding_dimension)),
            layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same'),
            layers.Reshape((784,))
        ])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Download the MNIST dataset
(x_train, _), (x_test, _) = mnist.load_data()

# Splitting the data into train and validation sets
train_size = int(0.8 * x_train.shape[0])
val_size = x_train.shape[0] - train_size
train_dataset, val_dataset = x_train[:train_size], x_train[train_size:]

# Convert datasets to float32 and normalize
train_dataset = tf.cast(train_dataset, tf.float32) / 255.0
val_dataset = tf.cast(val_dataset, tf.float32) / 255.0
test_dataset = tf.cast(x_test, tf.float32) / 255.0

# Reshape the datasets to (num_samples, 28, 28, 1)
train_dataset = tf.expand_dims(train_dataset, axis=-1)
val_dataset = tf.expand_dims(val_dataset, axis=-1)
test_dataset = tf.expand_dims(test_dataset, axis=-1)

train_dataset = tf.reshape(train_dataset, (-1, 28*28))
val_dataset = tf.reshape(val_dataset, (-1, 28*28))
test_dataset = tf.reshape(test_dataset, (-1, 28*28))

# Parameters
encoding_dimension = 78
epochs_param = 5
batch_size_param = 100

# Create and train the autoencoder
autoencoder = Autoencoder(encoding_dimension)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_dataset, train_dataset,
                epochs=epochs_param,
                batch_size=batch_size_param,
                shuffle=True,
                validation_data=(val_dataset, val_dataset)
                )

# Save the model
autoencoder.encoder.save('.\python\encoder.keras')

encoded_images = autoencoder.encoder(test_dataset).numpy()
decoded_images = autoencoder.decoder(encoded_images).numpy()

# Assuming x_test is reshaped to (num_samples, 28*28)
decoded_images_reshaped = decoded_images.reshape(-1, 28, 28)

# Visualize the original and reconstructed images
import matplotlib.pyplot as plt

# Choose a few indices to visualize
indices_to_visualize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Plot the original images
plt.figure(figsize=(9, 3))
for i, index in enumerate(indices_to_visualize):
    plt.subplot(2, len(indices_to_visualize), i + 1)
    plt.imshow(x_test[index].reshape(28, 28), cmap='gray')
    plt.axis('off')

# Plot the reconstructed images
for i, index in enumerate(indices_to_visualize):
    plt.subplot(2, len(indices_to_visualize), i + len(indices_to_visualize) + 1)
    plt.imshow(decoded_images_reshaped[index], cmap='gray')
    plt.axis('off')

plt.show()


# def triplet_loss(y_true, y_pred, alpha = 0.4):
#     """
#     Implementation of the triplet loss function
#     Arguments:
#     y_true -- true labels
#     y_pred -- python list containing three objects:
#             anchor -- the encodings for the anchor data
#             positive -- the encodings for the positive data (similar to anchor)
#             negative -- the encodings for the negative data (different from anchor)
#     Returns:
#     loss -- real number, value of the loss
#     """
#     anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
#     # distance between the anchor and the positive
#     pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    
#     # distance between the anchor and the negative
#     neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    
#     # compute loss
#     basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
#     loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
   
#     return loss