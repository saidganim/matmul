# See LICENSE.txt for license details.

NVCC = nvcc
CXX_FLAGS += -std=c++11 -O3 -Wall
PAR_FLAG = -fopenmp
#GENERATE = -D GENERATE
GENERATE = 
TIMING = -D TIMING

ifneq (,$(findstring icpc,$(CXX)))
	PAR_FLAG = -openmp
endif

ifneq ($(SERIAL), 1)
	CXX_FLAGS += $(PAR_FLAG)
endif

APPS = matmul transpose

.PHONY: all

all: $(APPS)

%.o: %.c 
	$(CXX) $(CXX_FLAGS) $(GENERATE) $(TIMING) -c  $^ $(LFLAGS)

mm_kernel.o: mm_kernel.cu
	$(NVCC) -O3 $^ -c

matmul: matmul.o mmio.o mm_kernel.o
	$(NVCC) matmul.o mmio.o mm_kernel.o -o $@


transpose: transpose.c mmio.c
	$(CXX) $(CXX_FLAGS) $(GENERATE) -o $@ $^ $(LFLAGS)

.PHONY: clean
clean:
	rm -f $(APPS)
	rm *.o
