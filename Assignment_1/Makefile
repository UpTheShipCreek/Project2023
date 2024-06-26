# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3

# Directories
LSH = ./lsh_folder
HYPERCUBE = ./hypercube_folder
CLUSTER = ./cluster_folder
INC = ./lib
OUT = ./out

# Source files
SRC_COMMON = $(INC)/random_functions.cpp $(INC)/io_functions.cpp $(INC)/metrics.cpp $(INC)/image_util.cpp $(INC)/hashtable.cpp
SRC_LSH = $(LSH)/src/main.cpp $(LSH)/src/lsh.cpp $(CLUSTER)/src/cluster.cpp
SRC_HYPERCUBE = $(HYPERCUBE)/src/main.cpp $(HYPERCUBE)/src/hypercube.cpp $(CLUSTER)/src/cluster.cpp
SRC_CLUSTER = $(CLUSTER)/src/main.cpp $(CLUSTER)/src/kMeans.cpp $(CLUSTER)/src/cluster.cpp $(LSH)/src/lsh.cpp $(HYPERCUBE)/src/hypercube.cpp

# Object files
OBJS_COMMON = $(SRC_COMMON:.cpp=.o)
OBJS_LSH = $(SRC_LSH:.cpp=.o)
OBJS_HYPERCUBE = $(SRC_HYPERCUBE:.cpp=.o)
OBJS_CLUSTER = $(SRC_CLUSTER:.cpp=.o)

# Include directories
INCLUDES = -I$(INC) -I$(LSH)/include -I$(HYPERCUBE)/include -I$(CLUSTER)/include

# Executable names
LSH_EXEC = lsh
HYPERCUBE_EXEC = cube
CLUSTER_EXEC = cluster

# Targets and rules
all: lsh cube cluster

lsh: $(LSH_EXEC)

hypercube: $(HYPERCUBE_EXEC)

cluster: $(CLUSTER_EXEC)

# I know that the includes are weird, couldn't get the Makefile to work otherwise
$(LSH_EXEC): $(OBJS_COMMON) $(OBJS_LSH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(LSH_EXEC)

$(HYPERCUBE_EXEC): $(OBJS_COMMON) $(OBJS_HYPERCUBE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(HYPERCUBE_EXEC)

$(CLUSTER_EXEC): $(OBJS_COMMON) $(OBJS_CLUSTER)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(CLUSTER_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_LSH) $(OBJS_HYPERCUBE) $(OBJS_CLUSTER) $(LSH_EXEC) $(HYPERCUBE_EXEC) $(CLUSTER_EXEC)