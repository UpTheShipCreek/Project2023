import argparse
import numpy as np
import tensorflow as tf

def write_mnist_images(filename, images_array, image_size):
    try:
        with open(filename, 'wb') as file:

            # Write the header information (16 bytes)
            magic_number = np.int32(7)
            file.write(magic_number.tobytes())

            number_of_images = np.int32(images_array.shape[0])
            file.write(number_of_images.tobytes())

            number_of_rows = np.int32(image_size)
            file.write(number_of_rows.tobytes())    

            number_of_columns = np.int32(1)
            file.write(number_of_columns.tobytes())

            print("Writing images... ")

            for image in images_array:
                pixel_bytes = image.astype(np.uint8)
                for pixel in pixel_bytes:
                    file.write(pixel.tobytes())

    except IOError:
        print("Error while writing to file")

    print("Done")

def read_mnist_images(filename):
    all_images = []

    try:
        with open(filename, 'rb') as file:
            # Skip the header information (16 bytes)
            file.seek(16)

            print("Reading images... ")

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

    except IOError:
        print("Error while opening file")
        return np.array([])

    print("Done")
    return np.array(all_images)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Apply an encoder to datasets.')
    parser.add_argument('dataset_path', help='Path to the dataset file.')
    parser.add_argument('queryset_path', help='Path to the query set file.')
    parser.add_argument('output_dataset_path', help='Path to the output dataset file.')
    parser.add_argument('output_queryset_path', help='Path to the output query set file.')
    args = parser.parse_args()

    # Load the datasets
    dataset = read_mnist_images(args.dataset_path)
    queryset = read_mnist_images(args.queryset_path)

    

    # Load the encoder
    encoder = tf.keras.models.load_model('./python/encoder.keras')

    # Apply the encoder to the datasets
    encoded_dataset = encoder.predict(dataset)
    encoded_queryset = encoder.predict(queryset)

    print("Intial shapes: ", dataset.shape, queryset.shape)
    print("Encoded shapes: ", encoded_dataset.shape, encoded_queryset.shape)
    _, image_size = encoded_queryset.shape

    write_mnist_images(args.output_dataset_path, encoded_dataset, image_size)
    write_mnist_images(args.output_queryset_path, encoded_queryset, image_size)

if __name__ == '__main__':
    main()