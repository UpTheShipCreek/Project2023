# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3

# Directories
INC = ./lib
OUT = ./out

# Source files
SRC_COMMON = $(INC)/graph.cpp $(INC)/random_functions.cpp $(INC)/io_functions.cpp $(INC)/metrics.cpp $(INC)/image_util.cpp $(INC)/hashtable.cpp $(INC)/lsh.cpp

# Object files
OBJS_COMMON = $(SRC_COMMON:.cpp=.o)

# Include directories
INCLUDES = -I$(INC)

# Executable name
GNNS_EXEC = gnns

# Targets and rules
all: gnns

gnns: $(GNNS_EXEC)

$(GNNS_EXEC): $(OBJS_COMMON)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(GNNS_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(GNNS_EXEC)
