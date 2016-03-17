# project configuration
LIB_GLUT_LINUX = -lGL -lGLU -lglut
LIB_GLUT_MAC   = -framework OpenGL -framework GLUT
CC             = g++
LD_FLAGS       = $(LIB_GLUT)
CC_FLAGS       = -Wall -Wno-deprecated-declarations -std=c++11 -Ofast -funroll-loops
EXEC           = digitscanner

# project structure
BUILD_DIR = build
BIN_DIR   = bin
SRC_DIR   = src

# libs and headers subfolders lookup
INCLUDE = -I$(SRC_DIR)
SRC     = $(wildcard $(SRC_DIR)/*.cpp)
OBJ     = $(patsubst $(SRC_DIR)/%.cpp, $(BUILD_DIR)/%.o, $(SRC))

# sourcefile subfolders lookup
VPATH = $(SRC_DIR)

# entry point
default:
	@echo "You need to specify the system you are building on. Possibilities:"
	@echo "  'make linux'"
	@echo "  'make mac'"

linux: lib_linux make_dir $(BIN_DIR)/$(EXEC)

mac: lib_mac make_dir $(BIN_DIR)/$(EXEC)

lib_linux:
	$(eval LD_FLAGS = $(LIB_GLUT_LINUX))

lib_mac:
	$(eval LD_FLAGS = $(LIB_GLUT_MAC))

make_dir:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# create binary
$(BIN_DIR)/$(EXEC): $(OBJ)
	$(CC) -o $@ $^ $(LD_FLAGS)

# objects
$(BUILD_DIR)/main.o: main.cpp DigitScanner.hpp Window.hpp Arguments.hpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<

$(BUILD_DIR)/Exception.o: Exception.cpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<

$(BUILD_DIR)/Window.o: Window.cpp Window.hpp GLUT.hpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<

$(BUILD_DIR)/Arguments.o: Arguments.cpp Arguments.hpp
	$(CC) $(INCLUDE) $(CC_FLAGS) -o $@ -c $<

clean:
	rm $(BUILD_DIR)/*.o
	rm -r $(BUILD_DIR)
