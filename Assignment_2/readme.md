# Software Development for Algorithmic Problems 2023-24

## Assignment 2: Graph-theoretic nearest neighbor search in C/C++

### Team 38
Krypotos Christos **sdi1700063**

Panagiotopoulos Georgios **sdi1700113**

This project offers two approximate algorithms for k-nearest-neighbor and range search using LSH/Hypercube projections. It also includes an implementation of k-means++ with MacQueen, allowing you to choose between approximate methods or the traditional Lloyd's algorithm.

The dataset being used is the MNIST dataset

## Table of Contents
- [Structure](#structure)
- [Compilation and Cleanup](#compilation-and-cleanup)
- [Usage](#usage)
    - [Input](#input)
    - [Calls](#calls)
    - [Output](#output)
- [Programs](#programs)
    - [Parameters](#parameters)
        - [Approximate Methods](#approximate-methods)
        - [Clustering](#clustering)
    - [Implementation Details](#implementation-details)
        - [General](#general)
        - [LSH](#lsh)
        - [Hypercube](#hypercube)
        - [Cluster](#cluster)

## Structure 
- **in**
- **lib**
- **out**
- **plots**
- **src**
- algorithm_comparisons
- Makefile 
- readme

## Compilation and Cleanup

    make
    make clean

## Usage

The project already contains two datasets, the a 5k sample of the MNIST training dataset and the testing datset named respectively:

    ./in/input.dat
    ./in/query.dat

The query dataset contains 10k images, but in our main we have reduced the amount of max queries to only 10, for speed. 
However, if one wishes to test a larger amount of queries, they can modify the ```QUERY_LIMIT``` in the ```main.cpp```.

The user is free to provide his own dataset as he pleases, assuming he gives the correct path and the files follow the same format as the MNIST ones.

### Calls
**GNNS**

    ./graph_search -d ./in/input5k.dat -q ./in/query.dat -k 100 -E 100 -R 400 -N 10 -m 1 -o ./out/gnns.out

**MRNG**

    ./graph_search -d ./in/input5k.dat -q ./in/query.dat -l 200 -m 2 -o ./out/mrng.out

Additionally the user may choose to omit to directly enter any parameters other than the ```-m``` parameter which is required.

### Output
The results are saved in the ```*.out``` files with the respective names. The ```*.out``` files are located in the 
```./out``` folder, but the user is free to specify a path for the output, so long as it is valid. 

## Programs
### Parameters
The parameters and the algorithms of the programs are discussed in relative detail in the [algorithm comparisons section](./algorithm_comparisons.md).

### Implementation Details