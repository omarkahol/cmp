# define the Cpp compiler to use
CXX = g++-15

# define any compile-time flags
CXXFLAGS := -std=c++20 -O3 -fopenmp -mmacosx-version-min=14.0 -DNDEBUG
DEBUG_CXXFLAGS := -std=c++20 -g -O0 -fopenmp -mmacosx-version-min=14.0

# Dependency generation for incremental builds
CXXFLAGS += -MMD -MP
DEBUG_CXXFLAGS += -MMD -MP

# (verbosity removed) Commands are always shown

# define external includes
EIGEN    = $(HOME)/opt/eigen-3.4.0/
NLOPTINC = $(HOME)/opt/nlopt-2.7.1/include
SPDLOGINC   = $(HOME)/opt/spdlog/include
CMPINC = $(HOME)/opt/CMP++/include
OMPINC = $(HOME)/opt/openmp/include
MATPLOTLIB = $(HOME)/opt/matplotlib-cpp/
SELF = .
PYTHONINC = /opt/homebrew/Frameworks/Python.framework/Versions/3.13/include/python3.13
NUMPYINC = /opt/homebrew/lib/python3.13/site-packages/numpy/_core/include
SVMINC = $(HOME)/opt/libsvm
USR = /opt/homebrew/include

# define external libs
NLOPTLIB = $(HOME)/opt/nlopt-2.7.1/lib
CMPLIB = $(HOME)/opt/CMP++/lib
OMPLIB = $(HOME)/opt/openmp/lib
PYTHONLIB = /opt/homebrew/Frameworks/Python.framework/Versions/3.13/lib
SVMLIB = $(HOME)/opt/libsvm

# include specific paths
INCLUDES = -I$(EIGEN) -I$(NLOPTINC) -I$(SPDLOGINC) -I$(CMPINC) -I$(MATPLOTLIB) -I$(OMPINC) -I$(SELF) -I$(PYTHONINC) -I$(NUMPYINC) -I$(SVMINC) -I$(USR)

# include specific libraries
LIBS = -L$(NLOPTLIB) -L$(CMPLIB) -L$(OMPLIB) -L$(PYTHONLIB) -L$(SVMLIB)

# define library flags
LFLAGS = -lcmp -lnlopt -lomp -lpython3.13 -lsvm

# Object directories
OBJDIR = obj
OBJDEBUGDIR = obj_debug

# Automatically find all test files
TEST_SOURCES = $(wildcard *.cpp)
TEST_TARGETS = $(TEST_SOURCES:.cpp=)
TEST_OBJECTS = $(addprefix $(OBJDIR)/, $(TEST_SOURCES:.cpp=.o))
TEST_DEBUG_OBJECTS = $(addprefix $(OBJDEBUGDIR)/, $(TEST_SOURCES:.cpp=.o))
TEST_EXECUTABLES = $(addprefix out_, $(TEST_TARGETS))
TEST_DEBUG_EXECUTABLES = $(addprefix out_debug_, $(TEST_TARGETS))

# Create uppercase targets for VS Code compatibility
UPPERCASE_TARGETS = $(shell echo $(TEST_TARGETS) | tr '[:lower:]' '[:upper:]')

# Default target builds all tests
all: $(TEST_EXECUTABLES)

# Debug target for individual tests
debug_%:
	@target_name=$$(echo $* | tr '[:upper:]' '[:lower:]'); \
	mkdir -p $(OBJDEBUGDIR); \
	$(CXX) $(DEBUG_CXXFLAGS) $(INCLUDES) -c $${target_name}.cpp -o $(OBJDEBUGDIR)/$${target_name}.o; \
	$(CXX) $(DEBUG_CXXFLAGS) $(INCLUDES) -o out_debug_$$target_name $(OBJDEBUGDIR)/$${target_name}.o $(LFLAGS) $(LIBS)

# Pattern rule to build executables from object files
out_%: $(OBJDIR)/%.o | $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $< $(LFLAGS) $(LIBS)

# Pattern rule to compile source files to object files
$(OBJDIR)/%.o: %.cpp | $(OBJDIR)
	mkdir -p $(OBJDIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Create lowercase target aliases so `make <test>` builds `out_<test>`
$(TEST_TARGETS):
	@$(MAKE) out_$@

# Include generated dependency files (safe if they don't exist yet)
-include $(patsubst $(OBJDIR)/%.o,$(OBJDIR)/%.d,$(TEST_OBJECTS))

# Create object directory
$(OBJDIR):
	mkdir -p $(OBJDIR)

.SECONDARY: $(TEST_OBJECTS)
.PRECIOUS: $(TEST_OBJECTS)

# Create uppercase target aliases for VS Code tasks
# This creates targets like GRID -> out_grid, MCMC -> out_mcmc, etc.
$(UPPERCASE_TARGETS):
	@target_lower=$$(echo $@ | tr '[:upper:]' '[:lower:]'); \
	$(MAKE) out_$$target_lower

# Help and clean targets
help:
	@echo "Usage: make [target]"
	@echo
	@echo "Main targets:"
	@echo "  all               Build all release tests (out_*)"
	@echo "  out_<name>        Build release test and place object in obj/"
	@echo "  debug_<name>      Build debug test (out_debug_<name>)"
	@echo "  list              List available tests"
	@echo "  clean             Remove debug artifacts"
	@echo "  clean_all         Remove all build artifacts (obj/, out_*)"
	@echo
	@echo ""

clean:
	@echo "Cleaning debug artifacts..."
	rm -rf $(OBJDEBUGDIR) $(TEST_DEBUG_EXECUTABLES) $(TEST_EXECUTABLES) $(OBJDIR)

# Show available tests
list:
	@echo "Available tests:"
	@echo "$(TEST_TARGETS)" | tr ' ' '\n' | sort

.PHONY: all clean clean_all list $(UPPERCASE_TARGETS)