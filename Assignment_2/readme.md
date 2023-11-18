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
        - [MRNG](#mrng)

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
###Input

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

#### General
To implement the latest graph algorithms, we used the existing utilities from a prior assignment, alongside a newly developed graph class responsible for the overall graph structure. This graph class is responsible for the general graph structure but also incorporates two search algorithms. Additionally, a specialized monotonic graph class is derived from the general graph class, featuring its own distinct constructor implementation.

The graph basically consists of its nodes, which we save in a vector structure, and the out-edges of each node, which we also store in a vector structure. The way we relate each node to its out edges is through a map. This implementation suits our purpose since in both GNNS and MRNG we are using the graph to traverse through the nodes, so we only care about relating the outgoing edges to each node i.e. "Where can we go from here?" and not "How many paths lead to here?".

#### GNNS
This algorithm consists of an initialization of the graph index with an approximate method ```initialize_neighbours_approximate_method``` and the ```k_nearest_neighbor_search``` algorithm from the assigment's notes. 

The initialization_approximate_method is pretty straightforward; the approximate method calls the LSH-KNN for the designated number of out-edges (k) and saves them as the neighbors of the node in the graph structure.

The ```k_nearest_neighbor_search``` a direct implementation of the algorithm we were given; we chose a random node, go through its neighbors and maybe (depending on the greedy steps variable) jump on the neighbor we deem closest to our goal (through a metric, we are using the Eucledean in this assignment) to repeat the process. This whole process we repeat R (random restarts) times. 

One implemetation deviation from the algorithm we were given is the usage of a Priority Queue instead of a sort in the end (similar to how we implemented the KNN in LSH and Hypercube).

#### MRNG
This algorithm also consists of an initialization of the graph index and a KNN algorithm, although in this case, we have implemented it as a seperate class, with the initialization of the index taking place on its constructor. 

For the constructor we followed the algorithm that was given in the notes, again using a Priority Queue to manage the neighbors. 
We also decided that since we were already iterating through all the nodes in the constructor, to also iteratively construct the Centroid of the nodes (using the sum formula we had derived in the previous assingment for the MacQueen updating of the clusters), which is used to create the Navigating Node the MRNG-KNN algorithm.

The MRNG-KNN calls the Graph method of ```generic_k_nearest_neighbor_search``` and initializes it with the Navigating node. Again we followed faithfully the algorithm from the notes incorporating the Priority Queue again. The algorithm itself is very simple, it's effectiveness purely relying on the fact that the graph was constructed to be Monotonic and thus naturally leading the search to nodes that are close to our query.  
