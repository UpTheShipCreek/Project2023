import tensorflow as tf
from keras import layers, Model
from keras.datasets import mnist
from keras_tuner import RandomSearch

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
  def __init__(self, encoding_dimension, layers_param, layer_size, activation_func1, activation_func2, regularizer, nnfactor):
    super(Autoencoder, self).__init__()

    self.encoding_dimension = encoding_dimension
    self.nnfactor = nnfactor

    # Encoder
    encoder_layers = [layers.Reshape((28, 28, 1), input_shape=(784,))]

    for _ in range(layers_param-1):
      encoder_layers.append(layers.Conv2D(layer_size, (3, 3), activation=activation_func1, padding='same', activity_regularizer=tf.keras.regularizers.l1(regularizer)))
      encoder_layers.append(layers.MaxPooling2D((2, 2), padding='same'))
    encoder_layers.append(layers.Conv2D(encoding_dimension, (3, 3), activation=activation_func2, padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-4)))
    encoder_layers.append(layers.Flatten())
    encoder_layers.append(layers.Dense(encoding_dimension))

    self.encoder = tf.keras.Sequential(encoder_layers)

  def call(self, x):
      encoded = self.encoder(x)
      return encoded

  def train_step(self, data):
      data, _ = data  # Unpack the tuple
      k = data.shape[0] // self.nnfactor
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

          # Calculate the sum of the losses
          ktLossSum = tf.reduce_sum(losses)
          total_loss = ktLossSum / data.shape[0]

      grads = tape.gradient(total_loss, self.trainable_variables)
      self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

      return {
          "loss": total_loss,
      }

  def test_step(self, data):
      data, _ = data  # Unpack the tuple
      k = data.shape[0] // self.nnfactor

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

      # Calculate the sum of the losses
      ktLossSum = tf.reduce_sum(losses)
      total_loss = ktLossSum / data.shape[0]

      return {
          "loss": total_loss,
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
epochs = 1
batch_size = 150

def build_autoencoder(hp):
    encoding_dimension = hp.Int('encoding_dimension', min_value=18, max_value=78, step=10)
    layers = hp.Int('layers', min_value=1, max_value=5, step=1)
    layer_size = hp.Int('layer_size', min_value=32, max_value=132, step=20)
    activation_func1 = hp.Choice('activation_func1', values=['relu', 'leaky_relu'])
    activation_func2 = hp.Choice('activation_func2', values=['relu', 'sigmoid'])
    regularizer = hp.Float('regularizer', min_value=1e-6, max_value=1e-4, sampling='log')
    nnfactor = hp.Int('nnfactor', min_value=1, max_value=5, step=1)

    autoencoder = Autoencoder(encoding_dimension, layers, layer_size, activation_func1, activation_func2, regularizer, nnfactor)

    autoencoder.compile(optimizer='adam', loss=custom_loss_function)
    return autoencoder

tuner = RandomSearch(
    build_autoencoder,
    objective='val_loss',
    max_trials=2,
    directory='tuner',
    project_name='autoencoder'
)

tuner.search(train_dataset, train_dataset, epochs=epochs, batch_size=batch_size, validation_data=(val_dataset, val_dataset), callbacks=[tf.keras.callbacks.EarlyStopping('val_loss', patience=3)])
best_model = tuner.get_best_models(num_models=1)[0]

best_model.save_weights('.\python\pencoder.keras')