# Build every .cu in this folder into build/<name>. Usage: make [ARCH=sm_90] && ./build/ch02_vecadd
NVCC  ?= nvcc
ARCH  ?= native
SRCS  := $(wildcard *.cu)
BINS  := $(patsubst %.cu,build/%,$(SRCS))

all: $(BINS)

build/%: %.cu
	@mkdir -p build
	$(NVCC) -O2 -arch=$(ARCH) -o $@ $<

clean:
	rm -rf build

.PHONY: all clean
