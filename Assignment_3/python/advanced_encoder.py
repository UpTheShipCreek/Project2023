import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras import layers, Model, losses
from keras.datasets import mnist
import keras.backend as K
from annoy import AnnoyIndex
import tensorflow_probability as tfp

def tensor_get_nearest(data, k):
    # Calculate the pairwise distances
    pairwise_distances = tf.norm(tf.expand_dims(data, 1) - tf.expand_dims(data, 0), axis=-1)

    # Get the indices of the k+1 nearest neighbors for each vector
    _, indices = tf.math.top_k(-pairwise_distances, k=k+1)

    # Exclude the first column, which is the index of the vector itself
    indices = indices[:, 1:]

    return indices

def get_nearest(batch, vector_size):

    def to_numpy(tensor):
        return tensor.numpy()

    batch = tf.py_function(to_numpy, [batch], tf.float32)

    k = vector_size // 5

    index = AnnoyIndex(vector_size, 'euclidean')

    for i, imageVector in enumerate(batch):
        index.add_item(i, imageVector)

    index.build(n_trees=10)

    srList = []
    for imageVector in batch:
        sr = index.get_nns_by_vector(imageVector, k+1, include_distances=False)[1:]
        srList.append(sr)
    
    return srList

def spearman_rank_correlation(y_true, y_pred):
    y_true_rank = tf.cast(tf.argsort(y_true), dtype=tf.float32)
    y_pred_rank = tf.cast(tf.argsort(y_pred), dtype=tf.float32)
    return tfp.stats.correlation(y_true_rank, y_pred_rank, sample_axis=None, event_axis=None)

def spearman_rank_loss(y_true, y_pred):
    correlation = spearman_rank_correlation(y_true, y_pred)
    return 1. - correlation

def kendal_tau_loss(y_true, y_pred, k):
    subtracted = tf.subtract(y_true, y_pred)
    normalized = K.switch(K.not_equal(subtracted, 0), K.ones_like(subtracted), subtracted)

    print("Normalized shape: ", normalized.shape, "Vector size: ", k)
    disordered = tf.reduce_sum(normalized)

    return disordered/k


def batch_kendal_tau_loss(encoder, batch, vector_size):
  initialSR = get_nearest(batch, vector_size)

  encoded_batch = encoder(batch)

  reducedSR = get_nearest(encoded_batch, vector_size)

  pairSRs = list(zip(initialSR, reducedSR))

  loss_sum = 0
  for pair in pairSRs:
    loss_sum += (lambda f, p : f(p))(kendal_tau_loss, pair)

  return loss_sum/(2*vector_size)

class Autoencoder(Model):
    def __init__(self, encoding_dimension):
        super(Autoencoder, self).__init__()

        self.encoding_dimension = encoding_dimension

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(78, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-6)),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(156, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-6)),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(78, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-6)),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(encoding_dimension, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-5)),
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
    
    def train_step(self, data):
        data, _ = data  # Unpack the tuple
        vector_size = data.shape[1]
        k = data.shape[0] // 5
        with tf.GradientTape() as tape:
            initialSR = tensor_get_nearest(data, k)
            encoded = self.encoder(data)
            reducedSR = tensor_get_nearest(encoded, k)

            # Stack the two tensors along the last dimension
            pairSRs = tf.stack([initialSR, reducedSR], axis=-1)
            pairSRs = tf.transpose(pairSRs, perm=[0, 2, 1])

            # Define a function to calculate the Kendall tau loss for a pair of rankings
            def calculate_loss(pair):
                y_true = tf.cast(pair[0], tf.float32)
                y_pred = tf.cast(pair[1], tf.float32)
                
                loss = kendal_tau_loss(y_true, y_pred, k)
                return tf.cast(loss, tf.float32)


            # Use tf.map_fn to apply the function to each pair of rankings
            ktLosses = tf.map_fn(calculate_loss, pairSRs, dtype=tf.float32)

            # Calculate the sum of the Kendall tau losses
            ktLossSum = tf.reduce_sum(ktLosses)
            custom_loss_value = ktLossSum / data.shape[0]

            decoded = self.decoder(encoded)
            reconstruction_loss = tf.reduce_mean(tf.square(data - decoded))  # Mean Squared Error
            total_loss = (0.0005 * reconstruction_loss) + (0.995 * tf.cast(custom_loss_value, tf.float32))

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kendall_tau_loss": custom_loss_value,
        }


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
encoding_dimension = 36
epochs_param = 100
batch_size_param = 500

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