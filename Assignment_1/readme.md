lsh call: 
    ./lsh -d ./in/input.dat -q ./in/query.dat -k 4 -L 5 -o ./out/lsh.out -N 10 -R 1000.0

cube call:
    ./cube -d ./in/input.dat -q ./in/query.dat -k 14 -M 10 -p 2 -o ./out/cube.out -N 10 -R 1000.0

Hypercube Notes:
    I've taken M to mean the number of hypercube vertices that I need to check cause this at the time made a lot more sense.
    Since I've noticed that M means actually numbe of prospective ImageVectors so I'll change it if need be but this makes very little sense given the default values.
    Are we expecting that 14 vertices, which only covers the hamming distance one vertices btw, are to have ~0.7 ImageVectors on average, some of them having zero? Or are we implicitly admitting that we need neither a 14-dimensional hypercube nor a probe value of 2?
    
    I believe the M parameter is entirely unnecessarry but, if it is exist, at least the default value should be set at:
        ```M = Sum (#_of_dataset_images / 2^k) * k!/i!(k-i)!, i = 0 -> probes```
    i.e. the average number of elements in a bucket/vertex multiplied by the number of different vertexes we are expected to visit given the space and the probe value 
