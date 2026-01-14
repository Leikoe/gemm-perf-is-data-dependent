# Simple benchmarks reproducing the paper "DGEMM performance is data-dependent" and the blog post "Strangely, Matrix Multiplications on GPUs Run Faster When Given "Predictable" Data!"

## Reference materials
- https://arxiv.org/abs/1912.05381
- https://www.thonking.ai/p/strangely-matrix-multiplications

## Requirements
- python3 
- matplotlib
- make
- cuda install

## How to run
```shell
make
./cuda_bencher > gpu_model.csv
python main.py path/to/gpu_model.csv # produces `flops_vs_zeroed_bits_gpu_model.png`
```
