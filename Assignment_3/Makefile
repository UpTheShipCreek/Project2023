CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
INC = ./modules
SRC = ./src
OUT = ./out

SRC_COMMON = $(wildcard $(INC)/general/*.cpp $(INC)/graph/*.cpp $(INC)/hash/*.cpp $(INC)/cluster/*.cpp)
OBJS_COMMON = $(patsubst %.cpp, %.o, $(SRC_COMMON))

SRC_LSH = $(SRC)/lsh_test.cpp 
SRC_REDUCED_EXHAUST = $(SRC)/reduced_exhaust_test.cpp
SRC_COMPARISONS = $(SRC)/comparisons.cpp

OBJS_LSH = $(SRC_LSH:.cpp=.o)
OBJS_REDUCED_EXHAUST = $(SRC_REDUCED_EXHAUST:.cpp=.o)
OBJS_COMPARISONS = $(SRC_COMPARISONS:.cpp=.o)

INCLUDES = -I$(INC)/general -I$(INC)/graph -I$(INC)/hash -I$(INC)/cluster
LSH_EXEC = lsh
REDUCED_EXHAUST_EXEC = reduced_exhaust
COMPARISONS_EXEC = comparisons

all: lsh reduced_exhaust comparisons

lsh: $(LSH_EXEC)

reduced_exhaust: $(REDUCED_EXHAUST_EXEC)

comparisons: $(COMPARISONS_EXEC)

$(LSH_EXEC): $(OBJS_COMMON) $(OBJS_LSH)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(LSH_EXEC)

$(REDUCED_EXHAUST_EXEC): $(OBJS_COMMON) $(OBJS_REDUCED_EXHAUST)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(REDUCED_EXHAUST_EXEC)

$(COMPARISONS_EXEC): $(OBJS_COMMON) $(OBJS_COMPARISONS)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(COMPARISONS_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_LSH) $(OBJS_REDUCED_EXHAUST) $(OBJS_COMPARISONS) $(LSH_EXEC) $(REDUCED_EXHAUST_EXEC) $(COMPARISONS_EXEC)
