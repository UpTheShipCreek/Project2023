# Software Development for Algorithmic Problems 2023-24


## Assignment 3: Dimensionality reduction using Neural Networks and experimentation

### Team 38
Krypotos Christos **sdi1700063**

Panagiotopoulos Georgios **sdi1700113**

This project focuses on dimensionality reduction of the MNIST dataset using a trained encoder, which transforms the original 28x28 images into a 20x1 vector. The reduced-dimensional encoding is then used for approximate nearest neighbor methods and clustering.


## Table of Contents
- [Structure](#structure)
- [Compilation and Cleanup](#compilation-and-cleanup)
- [Usage](#usage)
    - [Input Datasets](#input-datasets)
    - [Calls](#calls)
    - [Output](#output)
- [Programs](#programs)
    - [Implementation Details](#implementation-details)
        - [General](#general)
        - [Handling the space correspondance](#handling-the-space-correspondance)
        - [Reduce.py script](#reducepy-script)

## Structure 
- **encoder**
    - encoder.keras
- **in**
- **modules**
    - **cluster**
        - cluster.cpp/h
        - kmeans.cpp/h
    - **general**
        - image_util.cpp/h
        - io_functions.cpp/h
        - metrics.cpp/h
        - random_functions.cpp/h
    - **graph**
        - graph.cpp/h
        - mrng.cpp/h
    - **hash**
        - approximate_methods.h
        - hashtable.cpp/h
        - hypercube.cpp/h
        - lsh.cpp/h
- **out**
- **plots**
- **src**
    - clustering.cpp
    - comparisons.cpp
- comparisons.md
- encoder_creation.md
- Makefile 
- readme.md
- reduce.py

## Compilation and Cleanup
For the clustering tests:

    make clustering 

and

    make comparisons

for the comparisons of the nearest neighbor algorithms.

Of course

    make 

compiles both executables and

    make clean

deletes the executables and the objective files.

## Usage
### Input Datasets
We have again included the two datasets `input.dat` and `query.dat` with the whole project.

### Calls
For the encoder:

    python3 reduce.py -d ./in/input.dat -q ./in/query.dat -od ./in/encoded_dataset.dat -oq ./in/encoded_queryset.dat


After running the encoder and making sure your files follow the exact conventions mentioned above, you can run:
    
    ./clustering ./in/input.dat ./in/encoded_dataset.dat 

or 

    ./comparisons ./in/input.dat ./in/query.dat ./in/encoded_dataset.dat ./in/encoded_queryset.dat 100

for the clustering and nearest neighbor comparison tests respectively. 

### Output 
The detailed output of the comparisons will be written in the `./out/comparison_details.out`

## Programs
For the new process of the creation of the encoder see the [encoder creation section](./encoder_creation.md) and for the results see the [comparisons section](./comparisons.md)

### Implementation Details

#### General
It was relatively easy to get our previous algorithms running with the reduced dimension, we only needed to add one extra parameter in our hashtable methods, carrying the information of the dimension. Of course we also needed a correspondance between the two spaces. Luckily we were already storing the index of each image in our `ImageVector` class, so that was simple to implement as well. 

#### Handling the space correspondance
For the correspondance we created another class, called `SpaceCorrespondance` (its implementation located in `image_util.cpp`) that is simply a map from an `int` to an `ImageVector`, int being the image's index and `ImageVector` being the corresponding image structure of the original space. Initially we had made it to be a map of `int` to coodinates (i.e. `vector<double>`) but we decided to use the whole `ImageVectors` for ease of use and because we could just make a copies of their pointers (we are always handling `ImagesVectors` with shared pointers).

So, the way this works in practice is that when we find our k-nearest-neighbors on the reduced space, we iterate though them, find their correspondance on the original space using their `ImageVector::get_number()` method and then calculate their distance from our original query. 

#### Reduce.py script
The reduction script is very straighforward:
1. The magic number we write at the start of the encoded.dat files is `7` 
2. The way we normalize the values is by making them all positive (since we are turning them to unsigned ints). This we do by simply adding the least negative value of all the images. Then we scale them so that the maximum possible distance of the values of the pixels is 255 (which is the max for uint8).
