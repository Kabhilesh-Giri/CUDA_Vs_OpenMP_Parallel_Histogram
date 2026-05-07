/*
# NAME: Kabhilesh Giri
*/

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <math.h>

#define MIN_VALUE 1
#define MAX_VALUE 100000

// Define pair struct
typedef struct pair {
    int first;
    int second;
} pair;

// CUDA Kernel: For each data point, find its bin and atomically increment
__global__ void kernel(int *device_dataset, pair *device_bin, int *device_histogram, int N, int binCount) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < N) {
        int val = device_dataset[tid];
        // Linear search over bins
        for (int b = 0; b < binCount; b++) {
            if (val >= device_bin[b].first && val <= device_bin[b].second) {
                atomicAdd(&device_histogram[b], 1);
                break;
            }
        }
    }
}

// Random data generator
void setDataSet(int *data_set, int N) {
    for (int i = 0; i < N; i++) {
        data_set[i] = MIN_VALUE + rand() % (MAX_VALUE - MIN_VALUE + 1);
    }
}

void cudaHostDeviceExecution(int N, int threadsPerBlock)
{
    int binCount = 10;

    int bin_low = 0, bin_high = 0;
    int bin_Width = MAX_VALUE / binCount;
    float milliseconds = 0.0;

    //CUDA Event to measure timing for computation
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CUDA launch configuration
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Allocate host memory
    int *data_set = (int*)malloc(N * sizeof(int));
    int host_histogram[binCount] = {0};
    pair bin[binCount];

    // Init dataset and bins
    setDataSet(data_set, N);
    for (int f = 0; f < binCount; f++) {
        bin_low = f * bin_Width + 1;
        bin_high = bin_low + bin_Width - 1;
        if (f == binCount - 1) bin_high = MAX_VALUE;
        bin[f].first = bin_low;
        bin[f].second = bin_high;
    }

    // Allocate device memory
    int *device_dataset, *device_histogram;
    pair *device_bin;
    cudaMalloc((void**)&device_dataset, N * sizeof(int));
    cudaMalloc((void**)&device_bin, binCount * sizeof(pair));
    cudaMalloc((void**)&device_histogram, binCount * sizeof(int));
    cudaMemset(device_histogram, 0, binCount * sizeof(int));

    // Copy inputs to device
    cudaMemcpy(device_dataset, data_set, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_bin, bin, binCount * sizeof(pair), cudaMemcpyHostToDevice);

    cudaEventRecord(start);
    // Run kernel
    kernel<<<blocksPerGrid, threadsPerBlock>>>(device_dataset, device_bin, device_histogram, N, binCount);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    // Copy result back
    cudaMemcpy(host_histogram, device_histogram, binCount * sizeof(int), cudaMemcpyDeviceToHost);

    cudaEventSynchronize(stop);
    // Print result
    printf("Histogram Results:\n");
    for (int i = 0; i < binCount; i++) {
        printf("Bin %2d [%6d - %6d]: %d\n", i, bin[i].first, bin[i].second, host_histogram[i]);
    }

    cudaEventElapsedTime(&milliseconds,start,stop);
    printf("Timing to compute for Histogram Program : %.4f ms\n", milliseconds);

    // Free memory
    cudaFree(device_dataset);
    cudaFree(device_bin);
    cudaFree(device_histogram);
    free(data_set);
}

int main() {

    // This Part of the code is to study the GPU Properties to fix the N size and ThreadsPerBlock Size

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);

    printf("Max Threads Dim: [%d, %d, %d]\n",
        prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);

    printf("Max Grid Size: [%d, %d, %d]\n",
        prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

    srand(time(NULL));

    int N;
    int threadsPerBlock[4] = {126, 256, 512, 1024};

    // To Iterate from 2^12 to 2^23
    for (int i = 0; i < 4; i++)
    {
        for (int j = 12; j <= 23 ; j++)
        {
            N = (int)pow(2,j);
            printf("Block Size : %d | Array Size : %d\n",threadsPerBlock[i],N);
            cudaHostDeviceExecution(N,threadsPerBlock[i]);
            printf("\n");
        }
    }
    return 0;
}
