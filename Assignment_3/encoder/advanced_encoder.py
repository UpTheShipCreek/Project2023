import tensorflow as tf
from keras import layers, Model, losses
from keras.datasets import mnist

def tensor_get_nearest(dataIni, dataEnc, k):

    # The meat and bones of the custom loss function
    # It finds the batch/nnfactor nearest neighbors in the original dataset and uses the indices to find their equivalent distances on the reduced dataset
    # Then it finds the average difference in the order of magnitude in the distances (multiple)
    # It then uses this multiple to bring the order of magnitude of the distances in the reduced set, closer to the real order of magnitude of the same distances
    # The loss is calculated as the mean squared error of the k nearest neighbors in the multiplied reduced set of distances with the real set of distances for each image of the batch

    # Initial Data
    pairwise_distances_ini = tf.norm(tf.expand_dims(dataIni, 1) - tf.expand_dims(dataIni, 0), axis=-1)
    valuesIni, indices = tf.nn.top_k(pairwise_distances_ini, k=k-1, sorted=True)
    # Exclude the first column, which is the index of the vector itself
    valuesIni = valuesIni[:, 1:]
    indices = indices [:, 1:]

    # Encoded Data
    pairwise_distances_enc = tf.norm(tf.expand_dims(dataEnc, 1) - tf.expand_dims(dataEnc, 0), axis = -1)
    valuesEnc = tf.experimental.numpy.take_along_axis(pairwise_distances_enc, indices, axis=1)

    multiple = tf.reduce_mean(tf.divide(valuesIni, valuesEnc))

    return valuesIni, valuesEnc, multiple

def custom_loss_function(y_true, y_pred):
    subtracted = y_true - y_pred

    loss = tf.reduce_max(tf.square(subtracted))

    return loss

class Autoencoder(Model):
    def __init__(self, encoding_dimension):
        super(Autoencoder, self).__init__()

        self.encoding_dimension = encoding_dimension

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(52, (3, 3), activation='relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(3e-6)),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(encoding_dimension, (3, 3), activation='relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(3e-6)),
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
      
    def calculations(self, data):
        data, _ = data  # Unpack the tuple
        k = data.shape[0] // 4
        with tf.GradientTape() as tape:
            encoded = self.encoder(data)
            initialSR, reducedSR, multiple = tensor_get_nearest(data, encoded, k)

            reducedSR = tf.multiply(reducedSR, multiple)

            pairSRs = tf.stack([initialSR, reducedSR], axis=-1)
            pairSRs = tf.transpose(pairSRs, perm=[0, 2, 1])

            def calculate_loss(pair):
                y_true = pair[0]
                y_pred = pair[1]

                loss = custom_loss_function(y_true, y_pred)
                return tf.cast(loss, tf.float32)

            losses = tf.map_fn(calculate_loss, pairSRs, dtype=tf.float32)

            # Calculate Custom Loss
            ktLoss = tf.reduce_mean(losses)

            # Calculate the Prediction Loss
            decoded = self.decoder(encoded)
            prediction_loss = tf.reduce_mean(tf.square(data - decoded))

            total_loss = (0.5 * prediction_loss) + (0.1 * ktLoss)

        grads = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {
            "custom_loss": ktLoss,
            "prediction_loss": prediction_loss,
            "loss": total_loss, 
        }

    def train_step(self, data):
        return self.calculations(data)

    def test_step(self, data):
        return self.calculations(data)


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
encoding_dimension = 70
epochs_param = 100
batch_size_param = 40

# Create and train the autoencoder
autoencoder = Autoencoder(encoding_dimension)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(train_dataset, train_dataset,
                epochs=epochs_param,
                batch_size=batch_size_param,
                shuffle=True,
                validation_data=(val_dataset, val_dataset),
                callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)]
                )

autoencoder.encoder.save('./encoder.keras')