# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

# Directories
LSH = ./lsh_folder
HYPERCUBE = ./hypercube_folder
INC = ./lib
OUT = ./out

# Source files
SRC_COMMON = $(INC)/random_functions.cpp $(INC)/io_functions.cpp $(INC)/metrics.cpp $(INC)/image_util.cpp $(INC)/hashtable.cpp
SRC_LSH = $(LSH)/src/main.cpp $(LSH)/src/lsh.cpp
SRC_HYPERCUBE = $(HYPERCUBE)/src/main.cpp $(HYPERCUBE)/src/hypercube.cpp

# Object files
OBJS_COMMON = $(SRC_COMMON:.cpp=.o)
OBJS_LSH = $(SRC_LSH:.cpp=.o)
OBJS_HYPERCUBE = $(SRC_HYPERCUBE:.cpp=.o)

# Include directories
INCLUDES = -I$(INC) -I$(LSH)/include -I$(HYPERCUBE)/include

# Executable names
LSH_EXEC = lsh
HYPERCUBE_EXEC = hypercube

# Targets and rules
all: lsh hypercube

lsh: $(LSH_EXEC)

hypercube: $(HYPERCUBE_EXEC)

# I know that the includes are weird, couldn't get the Makefile to work otherwise
$(LSH_EXEC): $(OBJS_COMMON) $(OBJS_LSH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(LSH_EXEC)

$(HYPERCUBE_EXEC): $(OBJS_COMMON) $(OBJS_HYPERCUBE)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(HYPERCUBE_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_LSH) $(OBJS_HYPERCUBE) $(LSH_EXEC) $(HYPERCUBE_EXEC)
