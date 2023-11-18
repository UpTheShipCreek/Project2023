# Software Development for Algorithmic Problems 2023-24

## Assignment 2: Graph-theoretic nearest neighbor search in C/C++

### Team 38
Krypotos Christos **sdi1700063**

Panagiotopoulos Georgios **sdi1700113**

This project offers two approximate algorithms for k-nearest-neighbors using a graph-theoretic approach.

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
    - [Implementation Details](#implementation-details)
        - [General](#general)
        - [GNNS](#gnns)
        - [MNRG](#hypercube)

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

The project already contains three datasets, the original MNIST training dataset, a ```5K``` reduced version of it and the testing datset named respectively:

    ./in/input.dat
    ./in/input5k.dat
    ./in/query.dat

The query dataset contains 10k images, but in our main we have reduced the amount of max queries to only 10, for speed. 
However, if one wishes to test a larger amount of queries, they can modify the ```QUERY_LIMIT``` in the ```main.cpp```.

The user is free to provide his own dataset as he pleases, assuming he gives the correct path and the files follow the same format as the MNIST ones.

### Calls
**GNNS**
    
For the small ```5K``` dataset:

    ./graph_search -d ./in/input5k.dat -q ./in/query.dat -k 10 -E 10 -R 4 -N 10 -m 1 -o ./out/gnns.out

For the MNIST ```60K``` training dataset, for a more impressive number of neighbors:

    ./graph_search -d ./in/input.dat -q ./in/query.dat -k 20 -E 20 -R 4 -N 20 -m 1 -o ./out/gnns.out

**MRNG**

For the small ```5K``` dataset (since the construction of the monotonic graph is very slow):

    ./graph_search -d ./in/input5k.dat -q ./in/query.dat -l 20 -N 10 -m 2 -o ./out/mrng.out

Additionally the user may choose to omit to directly enter any parameters other than the ```-m``` parameter which is required.

### Output
The results are saved in the ```*.out``` files with the respective names. The ```*.out``` files are located in the 
```./out``` folder, but the user is free to specify a path for the output, so long as it is valid. 

## Programs
### Parameters
The parameters and the algorithms of the programs are discussed in relative detail in the [algorithm comparisons section](./algorithm_comparisons.md).

### Implementation Details
