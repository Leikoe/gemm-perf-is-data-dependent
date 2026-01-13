import csv

import matplotlib.pyplot as plt

RESULTS_FILE = "cuda.csv"

if __name__ == "__main__":
    bits_zeroed = []
    tflops = []

    with open(RESULTS_FILE, mode='r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Header: BitsZeroed,AvgTimeMs,AvgTFLOPS
            bits_zeroed.append(int(row['BitsZeroed']))
            tflops.append(float(row['AvgTFLOPS']))

    plt.figure(figsize=(10, 6))
    plt.plot(bits_zeroed, tflops, marker='o', linestyle='-', color='b', label='Performance')

    plt.title('SGEMM Performance vs. Floating Point Entropy', fontsize=14)
    plt.xlabel('Number of LSB Bits Zeroed (N)', fontsize=12)
    plt.ylabel('Effective TFLOPS', fontsize=12)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xticks(range(0, 33, 2))

    output_file = 'flops_vs_bits.png'
    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")
