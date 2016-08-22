INCLUDE_FLAG = -I include -I .
LINK_FLAGS = -lopenblas -llapack -lm -lgomp -lstdc++
OPT = -O4
CXX = g++
CXXFLAGS = $(INCLUDE_FLAG) `Rscript -e 'Rcpp:::CxxFlags()'` $(OPT) -std=c++11 -fopenmp -I/opt/miniconda3/lib/R/library/RcppArmadillo/include
RLIB_FLAGS = -fPIC -shared
SRC_FILES = $(filter-out main.cpp, $(wildcard *.cpp))

mnmashr.so: obj
	R CMD SHLIB -o mnmashr.so $(addsuffix .o, $(notdir $(basename $(SRC_FILES)))) $(LINK_FLAGS)

obj:
	$(CXX) $(CXXFLAGS) $(RLIB_FLAGS) -c $(SRC_FILES)

.PHONY: clean

clean:
	rm -f *.so *.o
