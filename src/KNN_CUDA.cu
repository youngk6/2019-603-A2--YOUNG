//============================================================================
// Name        : KNN_CUDA.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
using namespace std;

__global__ void KNN(int *predictions, ArffData *dataset, int num_elements, int k_neighbors)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    // dataset, predictions, num_elements, k_neighbors are here...
    // solve each instance by looking at what is in various arrays at tid

}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }

    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    int num_elements = dataset->num_instances();

    // Allocate host memory
    int *h_predictions = (int *)malloc(num_elements * sizeof(int));

    // Allocate the device input vector A
    int *d_predictions;
    cudaMalloc(&d_predictions, num_elements * sizeof(int));

    // Copy the host input vectors A and B in host memory to the device input vectors in
    cudaMemcpy(d_predictions, h_predictions, num_elements * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // set k value (number of neighbors)
    int k = 5;
    KNN<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, dataset, num_elements, k);

    /*

    int* predictions = KNN(dataset, 5);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
	*/
}

