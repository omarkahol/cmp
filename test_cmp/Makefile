# define the Cpp compiler to use
CXX = g++-13

# define any compile-time flags
CXXFLAGS	:= -std=c++17 -Wl,-ld_classic -g -O3

# define external includes
EIGEN    = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
SPDLOGINC   = $(HOME)/opt/spdlog/include
CMPINC = $(HOME)/opt/CMP++/include

# define external libs
NLOPTLIB = $(HOME)/opt/nlopt-2.7.1/lib
CMPLIB = $(HOME)/opt/CMP++/lib

#include specific paths
INCLUDES = -I$(EIGEN) -I$(NLOPTINC) -I$(SPDLOGINC) -I$(CMPINC)

#include specific libraries
LIBS = -L$(NLOPTLIB) -L$(CMPLIB)

# define library flags
LFLAGS = -lcmp -lnlopt

# define target file
TARGET = main.cpp

#files
FILES = ../src/density.cpp ../src/gp.cpp ../src/grid.cpp ../src/io.cpp ../src/kernel.cpp ../src/mcmc.cpp ../src/optimization.cpp ../src/pdf.cpp

# define executable
OUTPUT = out

test: main.cpp 
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $(TARGET) -o $(OUTPUT).o
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $(OUTPUT) $(OUTPUT).o $(LFLAGS) $(LIBS)

test_full: main.cpp
	$(CXX) $(CXXFLAG) $(INCLUDES) $(TARGET) $(FILES) -o $(OUTPUT) $(LFLAGS) $(LIBS)
clean:
	rm out*
	rm *_samples.csv
	rm *pred.csv
