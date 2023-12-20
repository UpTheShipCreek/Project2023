CXX = g++
CXXFLAGS = -std=c++11 -Wall -Wextra -O3
INC = ./modules
SRC = ./src
OUT = ./out
SRC_COMMON = $(wildcard $(INC)/general/*.cpp $(INC)/graph/*.cpp $(INC)/hash/*.cpp $(INC)/cluster/*.cpp)
OBJS_COMMON = $(patsubst %.cpp, %.o, $(SRC_COMMON))
SRC_MAIN = $(SRC)/main.cpp
OBJS_MAIN = $(SRC_MAIN:.cpp=.o)
INCLUDES = -I$(INC)/general -I$(INC)/graph -I$(INC)/hash -I$(INC)/cluster
MAIN_EXEC = graph_search

all: graph_search

graph_search: $(MAIN_EXEC)

$(MAIN_EXEC): $(OBJS_COMMON) $(OBJS_MAIN)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $^ -o $(MAIN_EXEC)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(OBJS_COMMON) $(OBJS_MAIN) $(MAIN_EXEC)
