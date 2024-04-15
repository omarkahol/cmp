# define the Cpp compiler to use
CXX = g++-13

# define any compile-time flags  
CXXFLAGS	:= -std=c++17 -Wl,-ld_classic -O3 -g

# define external includes
EIGEN    = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
SPDLOGINC   = $(HOME)/opt/spdlog/include
SELF = ./include
USR = /opt/homebrew/include

# define external libs
NLOPTLIB = $(HOME)/opt/nlopt-2.7.1/lib

# Extrenals include files and folders
INCLUDES = -I$(EIGEN) -I$(NLOPTINC) -I$(SPDLOGINC) -I$(SELF) -I$(USR)

#include specific libraries
LIBS = -L$(NLOPTLIB)

# define library flags
LFLAGS = -lnlopt

# define the objects
OBJ = output/io.o output/pdf.o output/kernel.o output/optimization.o output/mcmc.o output/grid.o output/density.o output/gp.o output/finite_diff.o

all: io pdf kernel optimization mcmc grid density gp finite_diff staticlib docs
	@echo Executing 'all' complete
	@echo Documentation up to date

io: src/io.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/io.cpp -o output/io.o
	@echo Compiled io

pdf: src/pdf.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/pdf.cpp -o output/pdf.o
	@echo Compiled pdf

finite_diff: src/finite_diff.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/finite_diff.cpp -o output/finite_diff.o
	@echo Compiled finite_diff

kernel: src/kernel.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/kernel.cpp -o output/kernel.o
	@echo Compiled kernel

optimization: src/optimization.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/optimization.cpp -o output/optimization.o
	@echo Compiled optimization

mcmc: src/mcmc.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/mcmc.cpp -o output/mcmc.o
	@echo Compiled mcmc

grid: src/grid.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/grid.cpp -o output/grid.o
	@echo Compiled grid

density: src/density.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/density.cpp -o output/density.o
	@echo Compiled density

gp: src/gp.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/gp.cpp -o output/gp.o
	@echo Compiled gp

staticlib: $(OBJ)
	ar cr lib/libcmp.a $(OBJ)
	@echo Created static library

dynamiclib: $(OBJ)
	$(CXX) -dynamiclib -fPIC -o lib/libcmp.dylib $(OBJ) $(LFLAGS) $(LIBS)

docs: Doxyfile
	doxygen Doxyfile
	
clean:
	rm -rf output/*.o 
	rm -rf lib/*.a