import argparse
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

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


def main():
    # CMD arguments parsing
    parser = argparse.ArgumentParser(description='MNIST Autoencoder')
    parser.add_argument('-d','dataset_path', required=True, help='Path to the MNIST dataset')
    parser.add_argument('-q', '--queryset', required=True, help='Path to the query dataset')
    parser.add_argument('-od', '--output_dataset_file', required=True, help='Output file for the compressed dataset vectors')
    parser.add_argument('-oq', '--output_query_file', required=True, help='Output file for the compressed query vectors')

    args = parser.parse_args()

    #Load input data
    file = open(args.dataset_path, 'rb')
    # Load MNIST dataset
    train_images = load_images(file)
    
    #Load input data
    file = open(args.dataset_path, 'rb')
    # Load MNIST dataset
    train_images = load_images(file)

    train_images = train_images.reshape((train_images.shape[0], 28, 28, 1))
    test_images = test_images.reshape((test_images.shape[0], 28, 28, 1))