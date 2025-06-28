CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -DNDEBUG
TARGET = benchmark
SOURCES = benchmark.cpp

$(TARGET): $(SOURCES) eytzinger.hpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SOURCES)

clean:
	rm -f $(TARGET)

run: $(TARGET)
	./$(TARGET)

.PHONY: clean run