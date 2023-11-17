# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3

# Directories
INC = ./lib
MAIN = ./src
OUT = ./out

# Source files
SRC_COMMON = $(INC)/graph.cpp $(INC)/random_functions.cpp $(INC)/io_functions.cpp $(INC)/metrics.cpp $(INC)/image_util.cpp $(INC)/hashtable.cpp $(INC)/lsh.cpp $(INC)/hypercube.cpp $(INC)/mrng.cpp
SRC_MAIN = $(MAIN)/main.cpp

# Object files
OBJS_COMMON = $(SRC_COMMON:.cpp=.o)
OBJS_MAIN = $(SRC_MAIN:.cpp=.o)

# Include directories
INCLUDES = -I$(INC)

# Executable name
MAIN_EXEC = graph_search

# Targets and rules
all: graph_search

graph_search: $(MAIN_EXEC)

$(MAIN_EXEC): $(OBJS_COMMON) $(OBJS_MAIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(MAIN_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_MAIN) $(MAIN_EXEC)