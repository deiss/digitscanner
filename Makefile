# project third party
LIB_LIST     = glut
INCLUDE_LIST = glut
LIB_GLUT     = -lGL -lGLU lib/glut/GLUT
LD_FLAGSSSS  = -lglut -lGLU32

# compilator and liker flags
CC       = g++
LD_FLAGS = $(LIB_GLUT)
CC_FLAGS = -Wall -std=c++11
EXEC     = digitscanner

# project configuration
BUILD_DIR   = build
BIN_DIR     = bin
SRC_DIR     = src
LIB_DIR     = lib
INCLUDE_DIR = include

# subfolders lookup
LIB     = $(foreach lib, $(LIB_DIR)/$(LIB_LIST), $(addprefix -L, $(lib)))
INCLUDE = $(foreach include, $(INCLUDE_DIR)/$(INCLUDE_LIST), $(addprefix -I, $(include)))
SRC     = $(wildcard $(SRC_DIR)/*.cpp)
OBJ     = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC))

VPATH = $(SRC_DIR)

# entry point
all: make_dir $(BIN_DIR)/$(EXEC) clean

# directory 'build'
make_dir:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# create binary
$(BIN_DIR)/$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LIB) $(LD_FLAGS)

# remove *.o and 'build' directory
clean:
	rm $(BUILD_DIR)/*.o
	rm -r $(BUILD_DIR)

# objects
$(BUILD_DIR)/main.o: main.cpp DigitScanner.hpp Window.hpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<

# objects
$(BUILD_DIR)/Exception.o: Exception.cpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<

# objects
$(BUILD_DIR)/Window.o: Window.cpp Window.hpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<

# objects
$(BUILD_DIR)/Arguments.o: Arguments.cpp Arguments.hpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<
