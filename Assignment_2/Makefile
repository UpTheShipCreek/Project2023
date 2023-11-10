# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3

# Directories
INC = ./lib
GNNS = ./gnns_folder
MRNG = ./mrng_folder
OUT = ./out

# Source files
SRC_COMMON = $(INC)/graph.cpp $(INC)/random_functions.cpp $(INC)/io_functions.cpp $(INC)/metrics.cpp $(INC)/image_util.cpp $(INC)/hashtable.cpp $(INC)/lsh.cpp
SRC_GNNS = $(GNNS)/src/main.cpp
SRC_MRNG = $(MRNG)/src/mrng.cpp

# Object files
OBJS_COMMON = $(SRC_COMMON:.cpp=.o)
OBJS_GNNS = $(SRC_GNNS:.cpp=.o)
OBJS_MRNG = $(SRC_MRNG:.cpp=.o)

# Include directories
INCLUDES = -I$(INC)

# Executable name
GNNS_EXEC = gnns
MRNG_EXEC = mrng

# Targets and rules
all: gnns mrng

gnns: $(GNNS_EXEC)

mrng: $(MRNG_EXEC)

$(GNNS_EXEC): $(OBJS_COMMON) $(OBJS_GNNS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(GNNS_EXEC)

$(MRNG_EXEC): $(OBJS_COMMON) $(OBJS_MRNG)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(MRNG_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_GNNS) $(GNNS_EXEC) $(OBJS_MRNG) $(MRNG_EXEC)
