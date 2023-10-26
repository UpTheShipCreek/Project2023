lsh call: 
    ./lsh -d ./in/input.dat -q ./in/query.dat -k 4 -L 5 -o ./out/lsh.out -N 10 -R 1000.0

cube call:
    ./cube -d ./in/input.dat -q ./in/query.dat -k 14 -M 2000 -p 500 -o ./out/cube.out -N 10 -R 1000.0

cluster call:
    ./cluster -i ./in/input.dat -c ./cluster_folder/config/cluster.conf -o ./out/cluster.out -m Classic -complete   
    
    ./cluster -i ./in/input.dat -c ./cluster_folder/config/cluster.conf -o ./out/cluster.out -m LSH -complete 

    ./cluster -i ./in/input.dat -c ./cluster_folder/config/cluster.conf -o ./out/cluster.out -m Hypercube -complete 

    

Approximate Methods notes:
    1. Initially I had implemented the approximate searches assuming that it was okay to treat the queries as part of the dataset itself,
    afterall I could just append the vectors and load the whole dataset, including the queries. 
    
    This changed when I tried to implement the reverse assignment in K-means, cause reloading the LSH tables every time we assign a new 
    virtual centroid (virtual meaning that it is not part of the Images themselves), would be extremely slow and costly. 
    
    Thus I implemented a HashTable method called virtual_insert, where it is not really pushing the point into the tables but 
    instead returns the bucketId and it's comparison Id (for the querying trick) as if it was pushed into the tables.

    I was also forced to break the information hiding principle by implementing a method to calculate the Id of an image.

    2. I had also made the mistake of not returning the ImageVectors themselves in the approximate searches, since the assignment for those
    methods only asked for the distance and the number of the image in the file. But the ImageVectors themselves were very much needed when
    it came to implementing the Reverse assingment for kmeans, where we are calling range approximations again and again, updating the rest of the
    image vectors according to the results

    Still I haven't definitevely solves any of those issues, cause I was afraid of creating too many different conflicts at the same time.
    For the time being, I decided to implement another, third search method in the approximate methods, which is basically the original approximate_range method 
    with the two above issues resolved. 

    It also had the issue of not returning the ImageVectors ordered, necessarilly but think I can fix that easily by 
    explicitly defining the order of the priority_queue according to the distances (doubles) and ignoring the ImageVector pointers.

Image Utilities notes:
    I defined what it means to be for two ImageVectors to be equal and I also created hash function, in order to be able to use the unordered set.
    I am not sure if it is correct/good though, since I am using the image number on the input file but the virtual ImageVectors just get -1 instead.

Hypercube Notes:
    I've taken M to mean the number of hypercube vertices that I need to check cause this at the time made a lot more sense.
    Since I've noticed that M means actually numbe of prospective ImageVectors so I'll change it if need be but this makes very little sense given the default values.
    Are we expecting that 14 vertices, which only covers the hamming distance one vertices btw, are to have ~0.7 ImageVectors on average, some of them having zero? Or are we implicitly admitting that we need neither a 14-dimensional hypercube nor a probe value of 2?
    
    I believe the M parameter is entirely unnecessarry but, if it is exist, at least the default value should be set at:
        M = Sum (#_of_dataset_images / 2^k) * k!/i!(k-i)!, i = 0 -> probes
    i.e. the average number of elements in a bucket/vertex multiplied by the number of different vertexes we are expected to visit given the space and the probe value 
