CXX := c++
CXXFLAGS := -std=c++17 -Wall -Wextra -I.
TARGET := main
BUILD_DIR := .build

SOURCES := $(shell find . -name '*.cpp' -not -path './$(BUILD_DIR)/*' -not -path './build/*')
OBJECTS := $(patsubst ./%.cpp,$(BUILD_DIR)/%.o,$(SOURCES))

.PHONY: all run clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) -o $@

$(BUILD_DIR)/%.o: %.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -c $< -o $@

run: $(TARGET)
	./$(TARGET)

clean:
	rm -rf $(BUILD_DIR) $(TARGET)
