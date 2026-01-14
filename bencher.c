#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <cblas.h>
#include <time.h>
#include <string.h>
#include <cpuid.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <fcntl.h>
#include <hwloc.h>

#define L3_SIZE 3e6
#define CACHE_LINE_SIZE 64
#define MAT_SIZE 2048
#define NB_REPEAT 20 
#define NB_REPEAT_WARMUP 10
#define MAX_MASK_SIZE 64

// Parent binds to Core 0, Child binds to Core 1
#define PARENT_CORE 0
#define CHILD_CORE  1

// --- Shared Memory Structure for IPC ---
typedef struct {
    volatile int state; // 0 = IDLE, 1 = MEASURING, 2 = EXIT
    volatile unsigned long long freq_sum_khz;
    volatile unsigned long samples;
} monitor_shm_t;

// --- HWLOC Binding Helper ---
void bind_to_core(int core_id) {
    hwloc_topology_t topology;
    hwloc_topology_init(&topology);
    hwloc_topology_load(topology);

    hwloc_bitmap_t cpuset = hwloc_bitmap_alloc();
    hwloc_bitmap_set(cpuset, core_id);

    // strict binding
    if (hwloc_set_cpubind(topology, cpuset, HWLOC_CPUBIND_PROCESS) < 0) {
        perror("Failed to bind process");
        exit(EXIT_FAILURE);
    }

    hwloc_bitmap_free(cpuset);
    hwloc_topology_destroy(topology);
}

// --- Frequency Reading Helper ---
// Reads the current frequency of the TARGET core (the Parent)
unsigned long get_cpu_freq_khz(int core_id) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", core_id);
    
    FILE *f = fopen(path, "r");
    if (!f) return 0; // Fail silently or handle error
    
    unsigned long freq_khz = 0;
    if (fscanf(f, "%lu", &freq_khz) != 1) freq_khz = 0;
    fclose(f);
    return freq_khz;
}

// --- Existing Time Diff Helper ---
struct timespec diff_timespec(const struct timespec *time0, const struct timespec *time1) {
    struct timespec diff = {.tv_sec = time1->tv_sec - time0->tv_sec,
        .tv_nsec = time1->tv_nsec - time0->tv_nsec};
    if (diff.tv_nsec < 0) {
        diff.tv_nsec += 1000000000;
        diff.tv_sec--;
    }
    return diff;
}

double *alloc_matrix() {
    double *mat = (double *)calloc(MAT_SIZE * MAT_SIZE, sizeof(double));
    assert(mat);
    return mat;
}

double *alloc_random_matrix() {
    double *mat = (double*)malloc(MAT_SIZE * MAT_SIZE * sizeof(double));
    assert(mat);
    for (int i = 0; i < MAT_SIZE * MAT_SIZE; i++) {
        double rnd_val = (double)rand() / RAND_MAX;
        mat[i] = rnd_val;
    }
    return mat;
}

void clear_cache() {
    const size_t bytes_to_read = L3_SIZE * 2; 
    uint8_t *buffer = malloc(bytes_to_read);
    assert(buffer);
    for (size_t mem_idx = 0; mem_idx < bytes_to_read; mem_idx += CACHE_LINE_SIZE) {
        buffer[mem_idx] = (uint8_t)mem_idx;
    }
    free(buffer);
}

// Modified bench to accept shared memory pointer
struct timespec bench(double *A, double *B, double *C, int mask_size, monitor_shm_t *shm) {
    struct timespec start, end, diff;
    double total_time_sec = 0.0;

    long mask = (~0UL) << mask_size;
    for (int k = 0; k < MAT_SIZE * MAT_SIZE; k++) {
        ((long *) A)[k] &= mask;
        ((long *) B)[k] &= mask;
    }

    // Warmup (Not monitored)
    for (int i = 0; i < NB_REPEAT; i++) {
        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
            MAT_SIZE, MAT_SIZE, MAT_SIZE, 
            1.0, A, MAT_SIZE, B, MAT_SIZE, 
            0, C, MAT_SIZE);
    } 

    // Measurement Loop
    for (int i = 0; i < NB_REPEAT; i++) {
        memset(C, 0, MAT_SIZE * MAT_SIZE * sizeof(double));
        clear_cache();

        // Signal Monitor to Start
        shm->freq_sum_khz = 0;
        shm->samples = 0;
        shm->state = 1; // MEASURING
        __sync_synchronize(); // Memory barrier

        assert(clock_gettime(CLOCK_MONOTONIC, &start) == 0);

        cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, 
            MAT_SIZE, MAT_SIZE, MAT_SIZE, 
            1.0, A, MAT_SIZE, B, MAT_SIZE, 
            0, C, MAT_SIZE);

        assert(clock_gettime(CLOCK_MONOTONIC, &end) == 0);

        // Signal Monitor to Stop (temporarily between iterations, or you could keep it running)
        // Here we pause it to average strictly over the computation
        shm->state = 0; // IDLE
        __sync_synchronize();

        diff = diff_timespec(&start, &end);
        total_time_sec += (double)diff.tv_sec + (double)diff.tv_nsec / 1e9;
    }

    double avg_sec = total_time_sec / NB_REPEAT;
    struct timespec avg_ts;
    avg_ts.tv_sec = (time_t)avg_sec;
    avg_ts.tv_nsec = (long)((avg_sec - avg_ts.tv_sec) * 1e9);

    return avg_ts;
}

void print_progress(int current, int total) {
    float percentage = (float)current / total * 100.0;
    printf("\rProgress: %.1f%% (%d/%d)", percentage, current, total);
    fflush(stdout);
}

double to_gflops(struct timespec t) {
    double time_sec = (double)t.tv_sec + (double)t.tv_nsec / 1e9;
    double total_ops = 2.0 * (double)MAT_SIZE * (double)MAT_SIZE * (double)MAT_SIZE;
    return (total_ops / time_sec) / 1e9;
}

int main(int argc, char **argv) {
    // 1. Setup Shared Memory
    monitor_shm_t *shm = mmap(NULL, sizeof(monitor_shm_t), 
                              PROT_READ | PROT_WRITE, 
                              MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    shm->state = 0;

    // 2. Fork Process
    pid_t pid = fork();

    if (pid == 0) {
        // --- CHILD PROCESS (MONITOR) ---
        bind_to_core(CHILD_CORE);
        
        while (1) {
            int state = shm->state;
            if (state == 2) break; // Exit signal
            
            if (state == 1) { // Measuring
                unsigned long f = get_cpu_freq_khz(PARENT_CORE);
                if (f > 0) {
                    // Use atomic add just to be safe, though single writer usually ok
                    __sync_fetch_and_add(&shm->freq_sum_khz, f);
                    __sync_fetch_and_add(&shm->samples, 1);
                }
                // Busy wait or tiny sleep? 
                // Too much sleep = bad sampling. No sleep = resource contention (if on same core).
                // Since we are on different cores, we can poll fast.
                usleep(10); 
            } else {
                usleep(1000); // Sleep while waiting for parent
            }
        }
        exit(0);
    }

    // --- PARENT PROCESS (WORKER) ---
    bind_to_core(PARENT_CORE);

    char *base_name = "benchmark_results_";
    char *tag = "default";
    char *ext = ".csv";
    if (argc > 1) {
        tag = argv[1];
    }
    
    // Construct filename
    size_t full_file_name_length = strlen(base_name) + strlen(tag) + strlen(ext) + 1;
    char *full_file_name = calloc(full_file_name_length, sizeof(char));
    strcat(full_file_name, base_name);
    strcat(full_file_name, tag);
    strcat(full_file_name, ext);

    putenv("OMP_NUM_THREADS=1"); 

    FILE *fp = fopen(full_file_name, "w");
    if (fp == NULL) {
        perror("Error opening file");
        kill(pid, SIGKILL);
        return EXIT_FAILURE;
    }

    // Updated Header
    fprintf(fp, "BitsZeroed,AvgFreq_MHz,AvgTFLOPS\n");

    struct timespec cur;
    int total_steps = MAX_MASK_SIZE; 
    int current_step = 0;

    double *A = alloc_random_matrix();
    double *B = alloc_random_matrix();
    double *C = alloc_matrix();

    printf("Parent on Core %d, Child on Core %d\n", PARENT_CORE, CHILD_CORE);
    printf("Starting benchmark...\n");

    for (int j = 0; j < MAX_MASK_SIZE; j += 1) {
        // Run Bench (shm is passed to control the child)
        cur = bench(A, B, C, j, shm);
        
        // Calculate average frequency
        double avg_freq_mhz = 0.0;
        if (shm->samples > 0) {
            avg_freq_mhz = (double)shm->freq_sum_khz / shm->samples / 1000.0;
        }

        fprintf(fp, "%d,%.2f,%.6f\n", j, avg_freq_mhz, to_gflops(cur));
        printf("\nMask %d: %.2f MHz | %.6f GFLOPS\n", j, avg_freq_mhz, to_gflops(cur));
        
        print_progress(++current_step, total_steps);
    } 

    // Cleanup
    shm->state = 2; // Tell child to exit
    wait(NULL);     // Wait for child
    
    free(A);
    free(B);
    free(C);
    fclose(fp);
    
    printf("\nDone.\n");
    return EXIT_SUCCESS;
}
