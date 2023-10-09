# Compiler and compiler flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra

# Directories
SRCLSH = ./lsh/src
INC = ./lib

# Source files
SRCS = $(SRCLSH)/hyperplanes.cpp $(INC)/random_functions.cpp
# Add more source files if needed

# Object files
OBJS = $(SRCS:.cpp=.o)

# Include directories
INCLUDES = -I$(INC)

# Executable name
EXEC = hyperplanes

# Targets and rules
all: $(EXEC)

$(EXEC): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS) $(EXEC)
