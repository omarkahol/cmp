# define the Cpp compiler to use
CXX = g++-13

# define any compile-time flags  
CXXFLAGS	:= -std=c++17 -Wl,-ld_classic -O3 -g

# define external includes
OMPINC   = $(HOME)/opt/openmp/include
STCINC   = $(HOME)/opt/StochTk++/includes
EIGEN    = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
FDIFFINC = $(HOME)/opt/finite-diff/include
CUBINC   = $(HOME)/opt/cubature/include
SPDLOGINC   = $(HOME)/opt/spdlog/include
SELF = ./include

#Extrenals include files and folders
INCLUDES = -I$(EIGEN) -I$(STCINC) -I$(NLOPTINC) -I$(SPDLOGINC) -I$(SELF)

all: io pdf kernel optimization mcmc doe density density_opt staticlib docs
	@echo Executing 'all' complete
	@echo Documentation up to date

io: src/io.cpp include/io.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/io.cpp -o output/io.o
	@echo Compiled io

pdf: src/pdf.cpp include/pdf.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/pdf.cpp -o output/pdf.o
	@echo Compiled pdf

kernel: src/kernel.cpp include/kernel.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/kernel.cpp -o output/kernel.o
	@echo Compiled kernel

optimization: src/optimization.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/optimization.cpp -o output/optimization.o
	@echo Compiled optimization

mcmc: src/mcmc.cpp include/mcmc.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/mcmc.cpp -o output/mcmc.o
	@echo Compiled mcmc

doe: src/doe.cpp include/doe.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/doe.cpp -o output/doe.o
	@echo Compiled doe

density: src/density.cpp include/density.h
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/density.cpp -o output/density.o
	@echo Compiled density

density_opt: src/density_opt.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c src/density_opt.cpp -o output/density_opt.o
	@echo Compiled density_opt

staticlib: output/io.o output/pdf.o output/kernel.o output/optimization.o output/mcmc.o output/doe.o output/density.o output/density_opt.o
	ar cr lib/libcmp.a output/io.o output/pdf.o output/kernel.o output/optimization.o output/mcmc.o output/doe.o output/density.o output/density_opt.o
	@echo created static library

docs: Doxyfile
	doxygen Doxyfile
	
clean:
	rm -rf output/*.o 
	rm -rf lib/*.a