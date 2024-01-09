# Software Development for Algorithmic Problems 2023-24


## Assignment 3: Dimensionality reduction using Neural Networks and experimentation

### Team 38
Krypotos Christos **sdi1700063**

Panagiotopoulos Georgios **sdi1700113**

This project focuses on dimensionality reduction of the MNIST dataset using a trained encoder, which transforms the original 28x28 images into a 20x1 vector. The reduced-dimensional encoding is then used for approximate nearest neighbor methods and clustering.


## Table of Contents

## Structure 
- **encoder**
    - encoder.keras
    - reduce.py
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

## Compilation and Cleanup

    python3 ./encoder/reduce.py ./in/input.dat ./in/query.dat ./in/encoded_dataset.dat ./in/encoded_queryset.dat

## Usage
### Input Datasets

### Calls

### Output

## Programs
### Implementation Details