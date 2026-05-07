#########################################
# CUDA vs OpenMP Parallel Histogram
#########################################

CC_CUDA ?= nvcc
CC_OPENMP ?= gcc

CUDA_SRC := CUDA_Parallel_Histogram.cu
OPENMP_SRC := OpenMP_Parallel_Histogram.c

CUDA_TARGET := cuda_histogram
OPENMP_TARGET := openmp_histogram

.PHONY: all cudarun openmprun clean

all: $(CUDA_TARGET) $(OPENMP_TARGET)

#########################################
# CUDA Compilation
#########################################

$(CUDA_TARGET): $(CUDA_SRC)
	@echo "Compiling CUDA histogram..."
	@$(CC_CUDA) $(CUDA_SRC) -lm -o $(CUDA_TARGET)

cudarun: $(CUDA_TARGET)
	@echo "Running CUDA histogram..."
	@./$(CUDA_TARGET) $(ARGS)

#########################################
# OpenMP Compilation
#########################################

$(OPENMP_TARGET): $(OPENMP_SRC)
	@echo "Compiling OpenMP histogram..."
	@$(CC_OPENMP) $(OPENMP_SRC) -fopenmp -lm -o $(OPENMP_TARGET)

openmprun: $(OPENMP_TARGET)
	@echo "Running OpenMP histogram..."
	@./$(OPENMP_TARGET) $(ARGS)

#########################################
# Cleanup
#########################################

clean:
	@echo "Cleaning generated binaries..."
	@rm -f $(OPENMP_TARGET) $(CUDA_TARGET) output cudaoutput
