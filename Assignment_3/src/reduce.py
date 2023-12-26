import argparse
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

#Functions 
def load_images(file):
    file.seek(18)

    images = []

    try:
        # Read and process each image entry
        while True:
            # Read 784 bytes (28x28 image size)
            image_pixels = np.frombuffer(file.read(784), dtype=np.uint8)

            # Check if the read operation resulted in an empty byte string
            if not image_pixels.any():
                break

            # Convert pixel values to double (no normalization in this case)
            normalized_pixels = image_pixels.astype(float)

            # Create an array representing the image
            singleImage = np.array(normalized_pixels.tolist())

            images.append(singleImage)

    except IOError:
        print("Error while opening file")
        return np.array([])

    return np.array(images)  

#Function builds the autoencoder
def BuildAutoEncoder():
    #encoder
    inputImage = tf.keras.input(shape=(28,28,1))
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(inputImage)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(16, (3, 3), activation='relu')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    # Create the autoencoder model
    autoencoder = models.Model(input_img, encoded)  # Output is the encoded layer
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

    return autoencoder


# Function to compress and save vectors
def compress_and_save(images, output_file):
    compressed_vectors = autoencoder.predict(images)

    # Reshape to (num_samples, flattened_vector_size)
    compressed_vectors = compressed_vectors.reshape((compressed_vectors.shape[0], -1))

    # Save compressed vectors to file
    np.save(output_file, compressed_vectors)


def main():
    #CMD arguments parsing
    parser = argparse.ArgumentParser(description='MNIST Autoencoder')
    parser.add_argument('-d','dataset_path', required=True, help='Path to the MNIST dataset')
    parser.add_argument('-q', '--queryset', required=True, help='Path to the query dataset')
    parser.add_argument('-od', '--output_dataset_file', required=True, help='Output file for the compressed dataset vectors')
    parser.add_argument('-oq', '--output_query_file', required=True, help='Output file for the compressed query vectors')

    args = parser.parse_args()

    #Load input data
    file = open(args.dataset_path, 'rb')
    #Load MNIST dataset
    train_images = load_images(file)
    
    #Load input data
    file = open(args.dataset_path, 'rb')
    #Load MNIST dataset
    train_images = load_images(file)

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))
    
    #Build and train autoencoder
    autoencoder = BuildAutoEncoder()
    autoencoder.fit(train_images, train_images, epochs=10, batch_size=128, shuffle=True, validation_data=(test_images, test_images))

    # Compress and save the dataset vectors
    compress_and_save(train_images, args.output_dataset_file, autoencoder)

    # Load, compress, and save the query dataset vectors
    query_images = load_images(args.queryset)
    query_images = query_images.reshape((query_images.shape[0], 28, 28, 1))
    compress_and_save(query_images, args.output_query_file, autoencoder)
