# Creating the Encoder
## Table of Contents
- [Introduction](#introduction)
- [Custom loss function](#custom-loss-function)
    - [General](#general)
    - [The pseudo-algorithm](#the-pseudo-algorithm)
    - [The code](#the-code)
- [Training](#training)
    - [Our approach](#our-approach)
    - [Results](#results)
    - [The model](#the-model)

## Introduction
The creation of the encoder was the core of this third assingment. Obviously, it is very easy to train a simple auto-encoder that performs extremely well on the MNIST set, since the dataset itself is very simple. 

In the context of this assigment though, this was not enough; we also needed to ensure that the encoding was useful in itself, and that the decoder didn't just learn to successfully decode whatever weird compression the encoder was coming up with. 

Our first intuition was to just make the decoder as simple as possible, let the encoder do the heavy lifting. This proved to not be such a great strategy, probably because the decoder didn't give enough feedback to help the encoder capture any meaningful patterns. Nonetheless, we were still unwilling to allow the decoder to do the heavy lifting, so we decided to create a custom loss function that captured exactly the information we wanted to capture, namely whether or not the distances of the images in the new space retained their ratios. 

## Custom loss function
### General
Our initial idea was to create a vector of the indexes of the k-nearest neighbors for every image in every batch, before and after the encoding. Then we would calculate something like the Kendall-Tau coefficient for every pair of vectors and using this derrive a loss value. Using this loss along with the prediction loss, would ensure that not only the encoder captured enough information to properly recreate the original image, but also that the encoding was meaningful in itself. 

We didn't implement this exact algorithm due to a technical limitation (i.e. operations on indices not being automatically differentiable in tensorflow) but we tried to approximate a loss function of this form. The results were satisfactorily enough. 

### The pseudo-algorithm
The algorithm for each batch ended up looking something like this:
* for each batch:
    * encoded = encode(batch)
    * for image in batch
        * true_nearest_vector = find_k_nearest_neighbors and their indices
    * for encoded_image in encoded
        * should_be_nearest_vector = get k neighbors using the indices
    * for image/encoded_image in batch/encoded
        * factors = true_nearest_vector / should_be_nearest_vector
        * factor = average(factors)
        * all_factors += factor
    * average_factor = average(all_factors)
    * for encoded_image in encoded
        * approximate_true_nearest_vector = average_factor * should_be_nearest_vector
    * for image/encoded_image in batch/encoded
        * squared_error_vector = (true_nearest_vector - approximate_true_nearest_vector)^2
        * squared_error_sum_of_max += max(squared_error_vector)
    * return average(squared_error_sum_of_max)
    
For each batch, find the nearest neighbors of the images in both the encoded and the original spaces, see by which average factor the distances differ, then multiply the encoded distances by this factor so that they try to approximate the true distances and try to minimize the loss between the true distances and the approximations.

Of course since we are forced to work with tensors all the `for` calculations were done with matrices. 

### The code 
Here is how we actually implemented it


```python
import tensorflow as tf
from keras import layers, Model, losses

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
        encoder_layers = [layers.Reshape((28, 28, 1))]

        for _ in range(layers_param-1):
            encoder_layers.append(layers.Conv2D(layer_size, (3, 3), activation=activation_func1, padding='same', activity_regularizer=tf.keras.regularizers.l1(regularizer)))
            encoder_layers.append(layers.MaxPooling2D((2, 2), padding='same'))
        encoder_layers.append(layers.Conv2D(encoding_dimension, (3, 3), activation=activation_func2, padding='same', activity_regularizer=tf.keras.regularizers.l1(1e-4)))
        encoder_layers.append(layers.Flatten())
        encoder_layers.append(layers.Dense(encoding_dimension))

        self.encoder = tf.keras.Sequential(encoder_layers)

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
        data, _ = data
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

            # Calculate Custom Loss
            ktLoss = tf.reduce_mean(losses)

            # Calculate the Prediction Loss
            decoded = self.decoder(encoded)
            prediction_loss = tf.reduce_mean(tf.square(data - decoded))

            total_loss = (0.5 * prediction_loss) + (0.05 * ktLoss)

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
```

## Training 
### Our approach
What was left then, was to do hyperparameter searches in order to find the best set of parameters. In order to get the best out of every combination of parameters, we allowed a large number of epochs `(40)` but used an early stopping with a small tolerance `(3)`. We also run each hyperparameter search with a set number of batches.

After each search, we would take the encoder and use it in our algorithms. In order to be efficient, we decided that brute force nearest neighbor search on the encoded space should at least approximate the accuracy and performance of LSH in the original space. 

Sadly, there wasn't a clear correlation between the validation loss in training and the performance of algorithm in our nearest neighbor search, so we don't a nice graph to show you. That of course means that our approach didn't exactly capture what we needed it to capture, but it ended up working well enough. 

### Results 
In particular, with a `latent dimension = 78` our approached achieved a `MAF = 1.2` and a much better average approximation factor, whilst being `~5` slower than our best LSH. 
But of course need a better latent dimension. 
We managed to achieve a `MAF = 1.7` with a `latent dimension = 20`. This time the exhaustive search in the latent space was just `~2.5` times slower than our best LSH. This we achieved with the model shown below. 

### The model


```python
class Autoencoder(Model):
    def __init__(self, encoding_dimension):
        super(Autoencoder, self).__init__()

        self.encoding_dimension = encoding_dimension

        # Encoder
        self.encoder = tf.keras.Sequential([
            layers.Reshape((28, 28, 1)),
            layers.Conv2D(112, (3, 3), activation='leaky_relu', padding='same', activity_regularizer=tf.keras.regularizers.l1(6e-5)),
            layers.MaxPooling2D((2, 2), padding='same'),
            layers.Conv2D(encoding_dimension, (3, 3), activation='sigmoid', padding='same', activity_regularizer=tf.keras.regularizers.l1(6e-5)),
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
        data, _ = data 
        k = data.shape[0] // 5
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
```
