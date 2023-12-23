import argparse
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='MNIST Autoencoder')
    parser.add_argument('-d','dataset_path', required=True, help='Path to the MNIST dataset')
    parser.add_argument('-q', required=True, help='Path to the query set file.')
    parser.add_argument('-od', required=True, help='Path to the output dataset file.')
    parser.add_argument('-oq', required=True, help='Path to the output query set file.')
    args = parser.parse_args()

    #Load input data
    open(args.dataset_path)

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
        image = np.array(normalized_pixels.tolist())

        all_images.append(image)