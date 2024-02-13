# define external includes
OMPINC   = $(HOME)/opt/openmp/include
STCINC   = $(HOME)/opt/StochTk++/includes
EIGEN    = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
CUBINC   = $(HOME)/opt/cubature/include
SPDLOGINC   = $(HOME)/opt/spdlog/include
CMPINC = $(HOME)/opt/CMP++/include

# define external libs
OMPLIB   = $(HOME)/opt/openmp/lib
STCLIB   = $(HOME)/opt/StochTk++/lib
NLOPTLIB = $(HOME)/opt/nlopt-2.7.1/lib
CUBLIB   = $(HOME)/opt/cubature/lib
CMPLIB = $(HOME)/opt/CMP++/lib

# define the Cpp compiler to use
CXX = g++-13

# define any compile-time flags
CXXFLAGS	:= -std=c++17 -Wl,-ld_classic -O3 -g

#include specific paths
INCLUDES = -I$(STCINC) -I$(EIGEN) -I$(NLOPTINC) -I$(SPDLOGINC) -I$(CMPINC)

#include specific libraries
LIBS = -L$(STCLIB) -L$(NLOPTLIB) -L$(CMPLIB)

# define library flags
LFLAGS = -lcmp -lgp++ -lnlopt

# define target file
TARGET = main.cpp

# define executable
OUTPUT = out

test: main.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(TARGET) -o $(OUTPUT).o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(OUTPUT) $(OUTPUT).o $(LFLAGS) $(LIBS)
clean:
	rm out*
	rm *_samples.csv
	rm *pred.csv
