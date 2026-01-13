CFLAGS=-I/usr/local/cuda/include -arch=sm_80

all: cuda_bencher

cuda_bencher: bencher.cu
	nvcc bencher.cu -o cuda_bencher $(CFLAGS) -lcublas -lcurand

clean:
	rm baseline cublas
