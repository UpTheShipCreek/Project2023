# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3

# Directories
INC = ./lib
GNNS = ./gnns_folder
MRNG = ./mrng_folder
OUT = ./out

# Source files
SRC_COMMON = $(INC)/graph.cpp $(INC)/random_functions.cpp $(INC)/io_functions.cpp $(INC)/metrics.cpp $(INC)/image_util.cpp $(INC)/hashtable.cpp $(INC)/lsh.cpp $(INC)/hypercube.cpp
SRC_GNNS = $(GNNS)/src/main.cpp
SRC_MRNG = $(MRNG)/src/mrng.cpp
SRC_LSH = $(GNNS)/src/lsh.cpp
SRC_HYPERCUBE = $(GNNS)/src/hypercube.cpp

# Object files
OBJS_COMMON = $(SRC_COMMON:.cpp=.o)
OBJS_GNNS = $(SRC_GNNS:.cpp=.o)
OBJS_MRNG = $(SRC_MRNG:.cpp=.o)
OBJS_LSH = $(SRC_LSH:.cpp=.o)
OBJS_HYPERCUBE = $(SRC_HYPERCUBE:.cpp=.o)

# Include directories
INCLUDES = -I$(INC)

# Executable name
GNNS_EXEC = gnns
MRNG_EXEC = mrng
LSH_EXEC = lsh
HYPERCUBE_EXEC = hypercube

# Targets and rules
all: gnns mrng lsh hypercube

gnns: $(GNNS_EXEC)

mrng: $(MRNG_EXEC)

lsh: $(LSH_EXEC)

hypercube: $(HYPERCUBE_EXEC)

$(GNNS_EXEC): $(OBJS_COMMON) $(OBJS_GNNS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(GNNS_EXEC)

$(MRNG_EXEC): $(OBJS_COMMON) $(OBJS_MRNG)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(MRNG_EXEC)

$(LSH_EXEC): $(OBJS_COMMON) $(OBJS_LSH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(LSH_EXEC)

$(HYPERCUBE_EXEC): $(OBJS_COMMON) $(OBJS_HYPERCUBE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(HYPERCUBE_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_GNNS) $(GNNS_EXEC) $(OBJS_MRNG) $(MRNG_EXEC) $(OBJS_LSH) $(LSH_EXEC) $(OBJS_HYPERCUBE) $(HYPERCUBE_EXEC)
