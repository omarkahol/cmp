# Define the C++ compiler to use
CXX = g++-15

# Define any compile-time flags
CXXFLAGS := -std=c++20 -O3 -fopenmp -mmacosx-version-min=14.0 -DNDEBUG
DEBUG_CXXFLAGS := -std=c++20 -g -O0 -fopenmp -mmacosx-version-min=14.0

# Define external includes
EIGEN = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
SELF = ./include
USR = /opt/homebrew/include
OMPINC = $(HOME)/opt/openmp/include
SVMINC = $(HOME)/opt/libsvm

# Define external libs
NLOPTLIB = $(HOME)/opt/nlopt-2.7.1/lib
OMPLIB = $(HOME)/opt/openmp/lib
SVMLIB = $(HOME)/opt/libsvm

# External include files and folders
INCLUDES = -I$(EIGEN) -I$(NLOPTINC) -I$(SELF) -I$(USR) -I$(OMPINC) -I$(SVMINC)

# Include specific libraries
LIBS = -L$(NLOPTLIB) -L$(OMPLIB) -L$(SVMLIB)

# Define library flags
LFLAGS = -lnlopt -lomp -lsvm

# Define the objects
OBJ = $(patsubst src/%.cpp,output/%.o,$(wildcard src/*.cpp))
DEBUG_OBJ = $(patsubst src/%.cpp,output/debug_%.o,$(wildcard src/*.cpp))

all: debug release
	@echo All builds complete

debug: $(DEBUG_OBJ) debug_staticlib debug_dynamiclib
	@echo Debug build complete

release: $(OBJ) staticlib dynamiclib
	@echo Release build complete

output/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo Compiled $< with flags: $(CXXFLAGS)

output/debug_%.o: src/%.cpp
	$(CXX) $(DEBUG_CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo Compiled $< with debug flags

staticlib: $(OBJ)
	ar cr lib/libcmp.a $(OBJ)
	@echo Created static library 
	@echo  

debug_staticlib: $(DEBUG_OBJ)
	ar cr lib/libcmp_debug.a $(DEBUG_OBJ)
	@echo Created debug static library 
	@echo  

dynamiclib: $(OBJ)
	$(CXX) -Wl,-ld_classic -dynamiclib -fPIC -o lib/libcmp.dylib $(OBJ) $(LFLAGS) $(LIBS)
	@echo Created dynamic library 
	@echo 

debug_dynamiclib: $(DEBUG_OBJ)
	$(CXX) -Wl,-ld_classic -dynamiclib -fPIC -o lib/libcmp_debug.dylib $(DEBUG_OBJ) $(LFLAGS) $(LIBS)
	@echo Created debug dynamic library 
	@echo 

docs: Doxyfile
	doxygen Doxyfile

clean:
	rm -rf output/*.o
	rm -rf lib/*.a