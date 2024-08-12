# define the Cpp compiler to use
CXX = g++-14

# define any compile-time flags
CXXFLAGS	:= -std=c++17 -Wl,-ld_classic -O3 -fopenmp -march=native

# define external includes
EIGEN    = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
SPDLOGINC   = $(HOME)/opt/spdlog/include
CMPINC = $(HOME)/opt/CMP++/include
OMPINC = $(HOME)/opt/openmp/include

# define external libs
NLOPTLIB = $(HOME)/opt/nlopt-2.7.1/lib
CMPLIB = $(HOME)/opt/CMP++/lib
OMPLIB = $(HOME)/opt/openmp/lib

#include specific paths
INCLUDES = -I$(EIGEN) -I$(NLOPTINC) -I$(SPDLOGINC) -I$(CMPINC) -I$(OMPINC)

#include specific libraries
LIBS = -L$(NLOPTLIB) -L$(CMPLIB) -L$(OMPLIB)

# define library flags
LFLAGS = -lcmp -lnlopt -lomp

# define target file
TARGET = main.cpp

# define executable
OUTPUT = out

test: main.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(TARGET) -o $(OUTPUT).o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(OUTPUT) $(OUTPUT).o $(LFLAGS) $(LIBS)
clean:
	rm out*
