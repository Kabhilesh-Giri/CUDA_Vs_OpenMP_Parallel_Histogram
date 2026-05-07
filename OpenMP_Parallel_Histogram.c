/*
# NAME: Kabhilesh Giri
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define MIN_VALUE 1
#define MAX_VALUE 100000

// Define pair struct
typedef struct pair
{
    int first;
    int second;
} pair;

// Random data generator
void setDataSet(int *data_set, int N)
{
    for (int i = 0; i < N; i++)
    {
        data_set[i] = MIN_VALUE + rand() % (MAX_VALUE - MIN_VALUE + 1);
    }
}

// CPU Histogram computation using OpenMP
void openmpHistogramComputation(int N, int numThreads)
{
    int binCount = 10;
    int bin_low = 0, bin_high = 0;
    int bin_Width = MAX_VALUE / binCount;

    double startTime, endTime;

    // Allocate memory
    int *data_set = (int *)malloc(N * sizeof(int));
    int histogram[binCount];
    pair bin[binCount];

    for (int i = 0; i < binCount; i++)
        histogram[i] = 0;

    // Init dataset and bins
    setDataSet(data_set, N);
    for (int f = 0; f < binCount; f++)
    {
        bin_low = f * bin_Width + 1;
        bin_high = bin_low + bin_Width - 1;
        if (f == binCount - 1)
            bin_high = MAX_VALUE;
        bin[f].first = bin_low;
        bin[f].second = bin_high;
    }

    startTime = omp_get_wtime();

// Parallelize the histogram generation
#pragma omp parallel for num_threads(numThreads)
    for (int i = 0; i < N; i++)
    {
        int val = data_set[i];
        for (int b = 0; b < binCount; b++)
        {
            if (val >= bin[b].first && val <= bin[b].second)
            {
#pragma omp atomic
                histogram[b]++;
                break;
            }
        }
    }

    endTime = omp_get_wtime();

    // Print result
    printf("Histogram Results:\n");
    for (int i = 0; i < binCount; i++)
    {
        printf("Bin %2d [%6d - %6d]: %d\n", i, bin[i].first, bin[i].second, histogram[i]);
    }

    printf("Timing to compute for Histogram Program : %.4f ms\n", (endTime - startTime) * 1000);

    free(data_set);
}

int main()
{
    int N;
    int threadsPerBlock[4] = {126, 256, 512, 1024};

    for (int i = 0; i < 4; i++)
    {
        for (int j = 12; j <= 23; j++)
        {
            N = (int)pow(2, j);
            printf("Block Size : %d | Array Size : %d\n", threadsPerBlock[i], N);
            openmpHistogramComputation(N, threadsPerBlock[i]);
            printf("\n");
        }
    }

    return 0;
}
