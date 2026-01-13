CFLAGS=-I/usr/local/cuda/include -arch=sm_80

all: cuda_bencher

cpu_bencher: bencher.c
	gcc bencher.c -o cpu_bencher -march=native -lopenblas


cuda_bencher: bencher.cu
	nvcc bencher.cu -o cuda_bencher $(CFLAGS) -lcublas -lcurand

clean:
	rm baseline cublas
