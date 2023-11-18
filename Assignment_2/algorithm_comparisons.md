# Algorithm Comparisons
It is worth noting that the more approximations performed the largest the chance that the approximation factor falls into a large outlier. The following tests were performed with a relatively small number of queries but large number of times, then calculating the average max outlier, in order to get a better sense. 

## LSH

### General
For the comparisons between the four algorithms we chose to take as a baseline the performance of LSH, since it is the most popular amogst them and yields fairly good results. 

We already had a good idea as to what the optimal parameters for LSH on the MNIST dataset were from the previous Assignment. Nonetheless, we run a wide range of tests and found a set of parameters that performed a bit more consistenly that the one we were previously using. 

### K, The number of Hyperplanes of the Hashfunctions
We first run LSH with a variable ```K```, leaving all the other variables constant. Increasing the number of the hyperplanes beyond the default value (4) yielded no notable improvments in accuracy, neither any notable increase in time. This means that the new "cuts" in the space don't offer any new information, which of course was what we had noticed from the previous assignment. 

Above (6) hyperplanes, the buckets seem to become sparse, since there were many accounts of K-NN failing to even find ```K``` candidates, let alone good ones. 

Decreasing the number of hyperplanes, increases the accuracy a bit but at a time cost

We performed those tests keeping the values which seemed optimal to us in the first assignment, that is:

* ```L = 5```
* ```Window = 1500```
* ```Table Size = 3750```

![png](plots/output_2_0.png)
    


### L, The number of Hashtables
We knew that L is the variable that has the most impact, both in approximation and in time. A large L value essentially simulates an exhaustive search by creating so many hashtables that for every point, they include all possible neighbors. Thus passing through every hashtable which result in calculating every distance from every point to another. 

Again we performed the test for L keeping the values which seemed optimal to us in the first assignment, that is:

* ```K = 4```
* ```Window = 1500```
* ```Table Size = 3750```

![png](plots/output_4_0.png)
    


### W, The size of the window of the Hyperplane function
This value we had experimentally learned that it was very important, it was the difference between filling up the buckets with good approximations or not at all. There were plenty of times that we weren't getting any results because of a very low ```W``` value. We had learned that the optimal for our implementation, as well as the MNIST dataset was somewhere around ~1400.

* ```L = 5```
* ```K = 4```
* ```Table Size = 3750```
    
![png](plots/output_6_0.png)
    


### Table Size, i.e. the number of buckets of each Hashtable
This value is given as ```SizeOfDataset/2^n``` . Of course we quickly rejected the really large values.
Again the test were run with the rest of the variables constant:

* ```L = 5```
* ```K = 4```
* ```W = 1500```
    
![png](plots/output_8_0.png)
    


### Finding the optimal combination
We run many tests, where we fed the program random parameters from a certain pool of parameters that we though to be within the optimal range. Then we took our optimal parameters and tested them for the actual max factor (not the average of many tries), to make sure that even our worst outlier was approximated good enough (i.e. our max approximation factor was less than ```2``` at all times).

Below the results of our "stress test" are shown. As you can see, you'd need to run more than ```5000``` queries in a row for a strong a enough chance that just one of them will over-approximate the nearest neighbor by a factor of ```2```. And all that achieved with only a ```0.0016 seconds/query```, in contrast with the brute force ```0.05 seconds/query``` (~32 times faster).

The tests below were run for the values: 

* ```L = 6```
* ```K = 4```
* ```W = 1400```
* ```TableSize = 7500```

![png](plots/output_10_0.png)
    


## Hypercube

### General
Having LSH as a baseline we will try to find the best parameters of the Hypercube that immitate the results of LSH.
Also based on our empirical knowledge of how the algorithm performs, we have formed a set of default values as the based testing ground of each parameter alone.

### K, the dimensions of the Hypercube
We obviously don't want to dimensions to be too low, since that would mean that there are not enough buckets to adequetly differentiate between the datapoints. Of course the more dimensions the most accurate our approximations, but that also depends on the rest of parameters. 

Thinking of the edge cases, it's easy to see that for low dimensions we would need a high M parameter (that is the number of elemets we are allowed to check) but we could get away with relatively low probes (the amount of neighboring vertices we are allowed to visit) since the hypercube verteces/buckets would contain many elements.

On the other hand, thinking of a case where each bucket/vertex only holds one datapoint (i.e. high dimensionality hypercube), we would need as many probes as number of elements that we are allowed to check, and we could possibly get away with only checking just a bit more than the specified amount of our KNN and get a very good approximation. This of course has the disadvantage of being extremely slow.

For the test, the rest of parameters are set to:

* ```Probes = 500```
* ```M = 3000```

![png](plots/output_13_0.png)
    


### Probes
There is nothing much to say about the probes, other than the fact the higher dimensions require more probes to reap their full potential. The peculiar thing about those particular parameters of the hypercube is that they all forego the natural rules of the structure and instead represent parameters that are intuitive for us to think of, but at the cost of being disconnected with the problem itself. 

This fact is also demonstrated in our implementation of the Hypercube, where the parameter ```probes``` is turn into the much more seemless ```MaxHammingDistance``` which neatly corresponds with the structure of the hypercube itself, in contrast with the parameter ```probes```, carries no direct information about how far you are allowed to strafe from the initial vertex/bucket. 

For the test, the rest of parameters are set to:

* ```K = 14```
* ```M = 3000```

![png](plots/output_15_0.png)
    


### M (number of elements allowed to check)
```M``` is probably the least impressive of all the parameters. It's an entirely artificial parameter that only serves the purpose of forcibly stopping the search once a certain number of elements have been checked. Unlike the ```probes``` parameter which at least can be mapped to a certain hamming distance given a hypercube of set dimensions, ```M``` lacks even this type of correspondance. 

Nonetheless, it is indeed helpful as an easy to calibrate the speed/accuracy tradeoff.
The following is how the algorithm performs for different ```M``` values given our favourite values of the rest of the parameters:

* ```K = 14```
* ```Probes = 500```

![png](plots/output_17_0.png)
    


### Finding the optimal combination
The Hypercube becomes slow when trying to immitate the accuracy of LSH but nonetheless we found a parameter combination that was accurate without compromising the querying speed too much. 
Specifically the time per query that Hypercube needs to achieve consistent worst case approximations less than a factor of ```2``` is ```0.0051 seconds/query``` (~10 less than brute force).

Those parameters are:

* ```K = 11```
* ```Probes = 6000```
* ```M = 4000```

![png](plots/output_19_0.png)
    


## GNNS
### General
Initially we were suprised at the performance of GNNS, since it had the ability to give better results than LSH, with fast querying speed as well. It was when we started running some more thorough tests that we realized that GNNS had the capacity to be very innacurate, yielding some terrible approximations.

Of course it also had the potential to be both fast and accurate, and that potential was hidden within the ```R``` (random restarts parameter). With a bit of a larger value than the default one (which was set to one), GNNS can yield fairly accurate results with a good querying speed. 

In the tests below, we will be using LSH to fill up the index. After finding parameters that appear optimal, we will compare the two initialization methods of LSH and Hypercube, using their optimal parameters, to see if there will be any visible difference in the performance of GNNS.

### R (Random Restarts)
This parameter is basically acts as a de-randomizer. That's because GNNS relies on a random starting node, which can throw off the whole search by a lot in some cases. Giving it many restarts increases the chances that this random node will sooner or later end up in the correct "neighborhood", as it were. 

That of course comes with a price; since the random restarts in a sense restart the GNNS algorithm, they can end up slowing it down by a lot. 

Below is the impact of the ```R``` parameter keeping the rest of the parameters at a modest level. Notice that average time of a query is analogus to the number of random restarts; when the latter is doubled the first one is doubled as well. This is exactly the behavior we expected. 

When it comes to its relation with the Average Max Factor, we followed the "Elbow" method. Notice in the graph below that the improvment is very steep at the ```R = 4```, which of course is the value what we chose to stick to.

* ```k = 20```
* ```greedySteps = 10```
* ```E = 20```

![png](plots/output_22_0.png)
    


### k (Max outdegree)
Since the index is initialized with approximate methods, which are not particularly accurate given the task of a large "K" in the KNN. That's why we restricted the ```k``` values to a pool of relatively small numbers.

* ```R = 4```
* ```greedySteps = 10```
* ```E = k```

![png](plots/output_24_0.png)
    


### E (Expansions)
This parameter determines how many of those out edges we will be using to find. Since we are in control of both ```k``` and ```E```, we could easily omit this parameter and just use every out edge in our search, calibrating the ```k``` parameter accordingly instead. For the sake of consistency, we have included the results of KNN with varying ```E```, using the max ```k``` from our previous test, to further demonstrate this correspondance. 

* ```R = 4```
* ```k = 100```
* ```greedySteps = 10```

![png](plots/output_26_0.png)
    


### G (Greedy Steps)
This parameter controls how many steps we will take from our initial node. Initially, the only canidates for our KNN are the direct neighbors of our initial **random** node, which is not ideal. ```G``` allows us to find better suited candidates by letting us take ```G``` steps in the (greedy) "right" direction, *id est* it allows us to jump to the node that is "closest" to our query node.

The thing is, it doesn't seem to be doing its job very well, maybe because of the Eucledian metric, or maybe because the approximate methods don't make a good enough graph. It slows the down the queries without offering any improvements in our approximations.

* ```k = 20```
* ```R = 4```
* ```E = 20```

![png](plots/output_28_0.png)
    


### Finding the optimal combination
The tests gave us a pretty good idea about the combination of parameter that could make GNNS competitive with LSH.

When it comes to the comparison of GNN-LSH and GNNS-Hypercube, as we expected, GNN-LSH seems to create a better graph than GNNS-Hypercube, as showcased by the comparison below.

As for the quality of the approximation, we settled for a result very similar to that of the pure Hypercube, both in accuracy (albiet GNNS here is still a bit more volatile) and in querying speed, with the speed hitting the exact goal of a ~10 times faster querying speed than brute force (```0.004690 seconds/query``` in comparison to ```0.05 seconds/query``` of the brute force). 

* ```K = 20```
* ```R = 20```
* ```E = 20```
* ```G = 10```
    
![png](plots/output_30_0.png)
    


## MRNG

### General 
The search that we've been tasked to implement for our Monotonic Relative Neighborhood Graphs, basically only takes one parameter, similar to to the ```M``` parameter of our Hypercube; the number of nodes we are allowed to check before we stop the algorithm.

Of course the meat of this algorithm is the initialization, in creating monotonic "graph paths" as it were. This is also the most expensive step, the initialization being quite slow for large datasets. Thus we decided to initilize it with only half of the MNIST dataset in our tests. 

### l (Size of candate set)

This parameter is very straight forward. Again, just like we saw with ```M``` parameter in the Hypercube, the ```l``` parameter serves as direct way to calibrate the accuracy/speed tradeoff. Because we were working with a reduced version of the set, we shifted into thinking more about the ratio of ```l/SizeOfDataset``` than absolute numbers. 

The speed we were looking for per query, was no more than the concession that was forced upon us in the previous methods (Hypercube/GNNS), meaning no less than a ~10 times speedup from the brute force.

While MRNG displayed a better behaviour than GNNS, it is still lacking (as far as handling the worst case goes) in comparison to the Hypercube and of course, the star of the show, LSH. 

### Finding the optimal combination
There wasn't much to this other than testing for different values of ```l```. Below are the results of our tests. And since this is only one parameter, we could afford to directly test for max approximation factors, rather than taking the average of many attempts.

The way we approached it was straight forward. We pushed the ```l``` as far as it would go without resulting a speedup that was less that the factor of ~10 of the brute force, for this given dataset.

The ```l``` value we settled on was ```l = 200``` which is roughly ```0.7%``` of the size of the dataset. Of course since ```l``` results in an absolute increase in querying time, this percentage will not be optimal in datasets much smaller or much larger than ```60000``` but it may be a good baseline.

The speed needed, for the below accuracies showcased below, is ```0.0060 seconds/query```.
 
![png](plots/output_34_0.png)
    


## Comparisons of the algorithms using our (optimal) parameters
In the graphs below we compare the max approximation factor for all the algorithms given an increasing dataset size, as well as the speeds of the algorithms using the parameters that performed best in our tests.
    
![png](plots/output_36_0.png)
    


# Conclusions
In the conducted tests, Locality-Sensitive Hashing (LSH) emerged as the obviously superior algorithm, showcasing the highest level of performance. Similar results were achieved with the Hypercube and GNNS algorithms, which, although slower, demonstrated a good level of accuracy.

On the other hand MRNG, even though its initial results seemed satisfactory, a notable drawback surfaced in its inconsistency when confronted with edge cases.

In conclusion, while LSH stands out as the optimal choice in terms of overall performance, the Hypercube and GNNS algorithms provide viable alternatives with a small sacrifice in speed. The MRNG algorithm, while proficient in many cases, struggles to maintain a high levels of accuracy when comfronted with large query sets. 

But of course we can't think of those results independently of the particular dataset (MNIST) and metric (Eucledean), both of which might be indirectly putting the graph searches at a disadvantage. 
