# Compiler settings
CXX = g++
CXXFLAGS = -Wall -Wextra -std=c++2b

# Target executable
TARGET = myprogram

# Source files
SRCS = hello.cpp

# Object files
OBJS = $(SRCS:.cpp=.o)

# Build rules
all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(TARGET) $(OBJS)
