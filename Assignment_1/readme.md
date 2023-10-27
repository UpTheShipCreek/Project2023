# Software Development for Algorithmic Problems 2023-24

## Assignment 1: Vector search and clustering in C/C++

### Team 38
Krypotos Christos **sdi1700063**

Panagiotopoulos Georgios **sdi1700113**

This project offers two approximate algorithms for k-nearest-neighbor and range search using LSH/Hypercube projections. It also includes an implementation of k-means++ with MacQueen, allowing you to choose between approximate methods or the traditional Lloyd's algorithm.

The dataset being used is the Mnist dataset

## Table of Contents
- [Structure](#structure)
- [Compilation](#compilation)
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
- **cluster_folder**
    - config
    - include
    - src 
- **hypercube_folder**
    - include
    - src 
- **in**
- **lib**
- **lsh_folder**
    - include
    - src 
- **out**
- Makefile
- readme

## Compilation
**LSH**
    
    make lsh

**Hypercube**
    
    make cube

**Cluster**
    
    make cluster

To make **all three** at once

    make 

**Clean up**
    
    make clean

## Usage
### Input

The project already contains a two datasets, the Mnist training dataset and the testing datset named respectively 
    
    ./in/input.dat
    ./in/query.dat

The user is free to provide his own dataset as he pleases, assuming he gives the correct path and the files follow the same format as the Mnist ones.

### Calls
**LSH**
    
    ./lsh -d ./in/input.dat -q ./in/query.dat -k 4 -L 5 -o ./out/lsh.out -N 1 -R 1000.0

**Hypercube** (*for results similar to lsh, not using the default values*)
    
    ./cube -d ./in/input.dat -q ./in/query.dat -k 14 -M 2000 -p 500 -o ./out/cube.out -N 1 -R 1000.0

**Cluster** (*using the query.dat as a dataset since input.dat was too large*)

- **Classic** (Lloyd's)
    
   ```./cluster -i ./in/query.dat -c ./cluster_folder/config  cluster.conf -o ./out/cluster.out -m Classic -complete```

- **LSH**
    
    ```./cluster -i ./in/query.dat -c ./cluster_folder/config cluster.conf -o ./out/cluster.out -m LSH -complete```
- **Hypercube**
    
    ```./cluster -i ./in/query.dat -c ./cluster_folder/config/cluster.conf -o ./out/cluster.out -m Hypercube -complete```

Additionally the user can simply call the **LSH** and **Hypercube** without any initial parameters and answer the prompts to fill them in.

### Output
The results are saved in the ```*.out``` files with the respective names. The ```*.out``` files are located in the 
```./out``` folder, but the user is free to specify a path for the output, so long as it is valid. 

Include examples and usage instructions.

## Programs
### Parameters
#### Approximate Methods
- Both **LSH** and **Hypercube Projection** are approximate methods for solving the **K-Nearest-Neighbor** problem and as thus share two parameters

    - **N** is lthe number of neighbors we want to find 
    - **R** is the range of the search i.e. what's the max range we want to find neighbors in.

- **LSH** exclusive parameters
    - **k** is the number of hyperplanes our hashfunction will consist of
    - **L** is the number of hashtables we will be creating
- **Hypercube** exclusive parameters
    - **k**  here is the number of dimensions we will be projecting to
    - **M** is the max number of elements we will be visiting in our search
    - **probes** is the max number of vertices of the hypercube we will be visiting in our search 

#### Clustering
Clustering takes as input a configuration file which contains information about the parameters that should be used in **LSH** and **Hypercube**. The formatting is shown below

    number_of_clusters: <int> // K of K-medians
    number_of_vector_hash_tables: <int> // default: L=3
    number_of_vector_hash_functions: <int> // k of LSH for vectors, default: 4
    max_number_M_hypercube: <int> // M of Hypercube, default: 10
    number_of_hypercube_dimensions: <int> // k of Hypercube, default: 3
    number_of_probes: <int> // probes of Hypercube, default: 2


### Implementation Details
#### General
We tried to ensure the implementation was as modular as possible. Seeing how both **LSH** and **Hypercube** shared a lot of similarities we decided to make some shared code that both implementations would rely on. 

- **Approximate methods** is a abstract class that both **LSH** and **Hypercube** inherit from and which sets the frame for all their methods

- **Hashtable** is a library which of course implements the Hashtable utility given a certain hashfunction (another virtual function, defined in the Hashtable library that the hashfunctions of both methods inherit from), but also some other functions which are used by both methods such as the **hi** hyperplane functions.

- **Note[1]**
    Initially we had implemented the **approximate searches** assuming that it was okay to treat the queries as part of the dataset itself, afterall we could just append the vectors and load the whole dataset, including the queries.

    This changed when we tried to implement the **reverse assignment** in **kMeans**, cause reloading the **LSH** tables every time we assigned a new virtual centroid (virtual meaning that it is not part of the Images dataset themselves), would be extremely slow and costly.`

    Thus we implemented a **HashTable** method called **virtual_insert**, where it is not really pushing the point into the tables but 
    instead returns the **bucketId** and it's comparison **Id** (for the querying trick) as if it was pushed into the tables.

- **Image Utilities** is a library which contains the class/type on which the images from the dataset are saved on, the **ImageVector** class. This class contains two variables, a vector<double> for the coordinates, and a number assigned to each image as an index, when it is first read. It is also where we have implemented our **exhaustive search functions**.

- **IO Functions** contains the functions that are used for reading from the dataset and writing out the results.

- **Metrics** contains the **Metric** abstract class that all the metrics should inherit from. At the moment the only actual metric it contains is the **Eucledean distance**.

- **Random Functions** actually contains a class of random functions. The advantage of it being a class is that we don't need to ever think about initilizing the random devices, or whether or not we are initializing them again and again. They are initialized the first time the function is called, and all subsequent calls use them.

#### LSH
**LSH** is a class, that implements the LSH method. It in construction is creates a vector of hashtables, initialzed with the specific **gi** hashfunctions, which are generated randomly as linear combination of **k** random **hi** function. 
- **load_data** finds the appropriate bucket in each hashtable for every image given vector of **ImageVectors**, our custom class.
- **Note[2]** 
    We had also made the mistake of not returning the **ImageVectors** themselves in the approximate searches since the assignment for those methods only asked for the **distance** and the **number of the image** in the file. But the **ImageVectors** themselves were very much needed when it came to implementing the **reverse assingment** for **kMeans**, where we are calling **range approximations** again and again, updating the rest of the
    image vectors according to the results. 

    That is where we decided we would create new **approximate search** methods, in fear of creating too many fronts at the same time if we were to change the old ones.

- **approximate_k_nearest_neighbors_return_images** takes as input an **ImageVector** and a number **k** and using a priority queue, returns a vector containing pairs of distances and **ImageVectors**.

- **approximate_range_search_return_images** does the same thing for given **range** instead of a set number of neighbors but doesn't return the vector shorted according to the distances (as it was not needed for **kMeans**)

- **Note[3]** It is also important to note here, that all the comparisons between images, i.e. if two images are the same or not, are done based on pointers and not using the unreliable indexing of the **ImageVector** class or the slow coortinate comparison, even though the latter is how **operation==** is implemented for **ImageVectors** (an operation which was never used in the end)

#### Hypercube
**Hypercube** also uses the **Hashtable** class for its vertices, albeit with a different hashfunction. All the methods are largely the same as in **LSH**, with one major difference since here we have probes instead of different tables. 
- For one, there is no need for comparisons as every vertex of the **Hypercube** will contain different elements.
- There is also some interest in how the search in different vertices is being done; the method should be visiting the closest neighbors until reaching the number of specified **probes**.
- **Note[4]** at first we were confused at what the **probes** input was meant to be, cause of the displayed default values of the assignment. He had taken it to be the max hamming distance of a vertex we are allowed to visit. Thus we had implemented our search directly built on that assumption; the **get_probes** would find and return a DFS vector of the **bucket-Ids/vertex coordinates**, then we would loop through every one of those until we had reached the specified amount of visited **ImageVectors/projected-points**. 
Turns out the **probes** variable actually meant the maximum number of vertices we were allowed to visit. And so to adapt the new meaning into our previous implemetation we created the **calculate_number_of_probes_given_maximum_hamming_distance** which relates the previous meaning of probes into the new one.

#### Cluster
There are two main classes that we implemented for the needs of **kMeans** clustering, one being the **kMeans** class itself and the other being the **Cluster** class, which implements the cluster structure.

- **Cluster** defines a cluster by saving its **Centroid**
as well as a **vector of points** that it contains. Its are entirely limited on managing the cluster, **changing/setting** the centroid, **adding** points, **recalculating** centroid as the center mass of all the points, as well as methods that give us access to that imformation **get_centroid** and **get_points**

- **kMeans** is a little more intense. Of course it holds the vital infomration for the method, such as a **vector of clusters**, the **dataset** itself and the **number of clusters**. 
    - The constructor initializes the **centroids** using the **Initialization++** technique. This is done by artificially creating a biased range, as sum of the squares of the distances of every point to the already existing centroids, and then picking a random number uniformly from that range.
    - 