#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <climits>
#include <fstream>
#include <string>
#include <getopt.h>

using namespace std;

//The main function. Bet you didn't expect that!
int main(int argc, char **argv){
    int k = 14, M = 10, probes = 2, N = 1, R = 10000;                               //Default values for hypercube projection, if not altered by user
    int opt;
    extern char *optarg; 
    string givenInput, queryFile, givenOutput;
    int cmdNecessary = 0;

    while ((opt = getopt(argc, argv, "d:q:k:M:p:o:N:R:")) != -1){                   //Parse through (potential) command line arguments
        switch (opt) {
            case 'd':                                                               //Files
                givenInput = optarg;
                cmdNecessary++;
                break;
            case 'q':
                queryFile = optarg;
                cmdNecessary++;
                break;
            case 'o':
                givenOutput = optarg;
                cmdNecessary++;
                break;
            case 'k':                                                               //Parameters
                k = atoi(optarg);
            case 'M':
                M = atoi(optarg);
            case 'p':
                probes = atoi(optarg);
                break;
            case 'N':
                N = atoi(optarg);
                break;
            case 'R':
                R = atoi(optarg);
                break;
        }
    }

    if(cmdNecessary != 3){
        cout << "Program execution requires an input file, output file and a query file.";
        exit(-1);
    }
    else{
        cout << "Program will proceed with values: k = " << k << ",M = " << M << ",probes = " << probes << "N = " << N << "R = " << R << "\n";
    }

    //Input files parcing
}