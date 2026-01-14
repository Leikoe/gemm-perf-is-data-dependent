CFLAGS=-I/usr/local/cuda/include -arch=sm_80

all: cuda_bencher

CPU_CFLAGS += $(shell pkg-config --cflags hwloc openblas)
CPU_LIBS   += $(shell pkg-config --libs hwloc openblas)

cpu_bencher: bencher.c
	gcc bencher.c -o cpu_bencher -march=native $(CPU_CFLAGS) $(CPU_LIBS)


cuda_bencher: bencher.cu
	nvcc bencher.cu -o cuda_bencher $(CFLAGS) -lcublas -lcurand

clean:
	rm baseline cublas
