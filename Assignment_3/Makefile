CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
INC = ./modules
SRC = ./src
OUT = ./out

SRC_COMMON = $(wildcard $(INC)/general/*.cpp $(INC)/graph/*.cpp $(INC)/hash/*.cpp $(INC)/cluster/*.cpp)
OBJS_COMMON = $(patsubst %.cpp, %.o, $(SRC_COMMON))

SRC_COMPARISONS = $(SRC)/comparisons.cpp
SRC_CLUSTERING = $(SRC)/clustering.cpp

OBJS_COMPARISONS = $(SRC_COMPARISONS:.cpp=.o)
OBJS_CLUSTERING = $(SRC_CLUSTERING:.cpp=.o)

INCLUDES = -I$(INC)/general -I$(INC)/graph -I$(INC)/hash -I$(INC)/cluster
COMPARISONS_EXEC = comparisons
CLUSTERING_EXEC = clustering

all: comparisons clustering

lsh: $(LSH_EXEC)

reduced_exhaust: $(REDUCED_EXHAUST_EXEC)

comparisons: $(COMPARISONS_EXEC)

clustering: $(CLUSTERING_EXEC)

$(COMPARISONS_EXEC): $(OBJS_COMMON) $(OBJS_COMPARISONS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(COMPARISONS_EXEC)

$(CLUSTERING_EXEC): $(OBJS_COMMON) $(OBJS_CLUSTERING)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(CLUSTERING_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_COMPARISONS) $(OBJS_CLUSTERING) $(COMPARISONS_EXEC) $(CLUSTERING_EXEC)
