# Define the C++ compiler to use
CXX = g++-14

# Define any compile-time flags
CXXFLAGS := -std=c++17 -Wl,-ld_classic -O3

# Define external includes
EIGEN = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
SPDLOGINC = $(HOME)/opt/spdlog/include
SELF = ./include
USR = /opt/homebrew/include
OMPINC = $(HOME)/opt/openmp/include

# Define external libs
NLOPTLIB = $(HOME)/opt/nlopt-2.7.1/lib
OMPLIB = $(HOME)/opt/openmp/lib

# External include files and folders
INCLUDES = -I$(EIGEN) -I$(NLOPTINC) -I$(SPDLOGINC) -I$(SELF) -I$(USR) 

# Include specific libraries
LIBS = -L$(NLOPTLIB) -L$(OMPLIB)

# Define library flags
LFLAGS = -lnlopt -lomp

# Define the objects
OBJ = $(patsubst src/%.cpp,output/%.o,$(wildcard src/*.cpp))

all: $(OBJ) staticlib dynamiclib docs
	@echo Executing 'all' complete
	@echo Documentation up to date

output/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@
	@echo Compiled $<

staticlib: $(OBJ)
	ar cr lib/libcmp.a $(OBJ)
	@echo Created static library 
	@echo  

dynamiclib: $(OBJ)
	$(CXX) -Wl,-ld_classic -dynamiclib -fPIC -o lib/libcmp.dylib $(OBJ) $(LFLAGS) $(LIBS)
	@echo Created dynamic library 
	@echo 

docs: Doxyfile
	doxygen Doxyfile

clean:
	rm -rf output/*.o
	rm -rf lib/*.a