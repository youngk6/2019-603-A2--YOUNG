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

__global__ void KNN(int *predictions, float *dataset, int *classes, float *distance_calculations, int k_neighbors, int num_elements)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    /*float ** Arr_dist = new float*[num_elements];
    int ** Arr_classes = new int*[num_elements];
    for (int i = 0; i < k_neighbors; i++){
    	Arr_dist[i] = (float *)malloc(k_neighbors * sizeof(float));
    	Arr_classes[i] = (int *)malloc(k_neighbors * sizeof(int));
    }*/
    //for loop to look at other instances in array
    for(int i = 0; i < num_elements; i++){
    	if(i == tid) continue;

    	/*for(int k = 0; k < dataset->num_attributes() - 1; k++){ // compute distance between two instances
    		float diff = dataset->get_instance(tid)->get(k)->operator float() - dataset->get_instance(i)->get(k)->operator float();
    		// this gets i, gets feature k, looks at distance from other instance j to k
    		distance_calculations[tid] += diff * diff;
    	}
    	distance_calculations[tid] = sqrt(distance_calculations[tid]);*/

    }

}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }

    // get dataset
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();

    // create 1D array of the data (without classes)
    int num_elements = dataset->num_instances();
    int total_attributes = dataset->num_attributes();

    float *h_arr_data = (float *)malloc(num_elements * total_attributes * sizeof(float));
    int count = 0;
    for (int i = 0; i < num_elements; i++){
    	for (int j = 0; j < total_attributes - 1; j++){
    		h_arr_data[count] = dataset->get_instance(i)->get(j)->operator float();
    		count++;
    	}
    }

    // create 1D array of classes for each instance
    count = 0;
    int *h_arr_class_data = (int *)malloc(num_elements * sizeof(int));

    for(int i = 0; i < num_elements; i++){
    	h_arr_class_data[count] = dataset->get_instance(i)->get(dataset->num_attributes()-1)->operator long int();
    	count++;
    }

    // start clock
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // Allocate other host memory
    int *h_predictions = (int *)malloc(num_elements * sizeof(int));
    float * h_distance_calculations = (float *)calloc(num_elements, sizeof(float));

    // Allocate the device input vector A
    int *d_predictions;
    float *d_distance_calculations;
    int *d_arr_class_data;
    float *d_arr_data;

    cudaMalloc(&d_predictions, num_elements * sizeof(int));
    cudaMalloc(&d_distance_calculations, num_elements *sizeof(float));
    cudaMalloc(&d_arr_class_data, num_elements * sizeof(int));
    cudaMalloc(&d_arr_data, num_elements * total_attributes * sizeof(float));

    // Copy the host input vectors A and B in host memory to the device input vectors in
    cudaMemcpy(d_predictions, h_predictions, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance_calculations, h_distance_calculations, num_elements *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_class_data, h_arr_class_data, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_data, h_arr_data, num_elements * total_attributes * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // set k value (number of neighbors)
    int k = 5;
    KNN<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_arr_data, d_arr_class_data, d_distance_calculations, k, num_elements);

    cudaMemcpy(h_predictions, d_predictions, num_elements * sizeof(int), cudaMemcpyDeviceToHost);

    /*

    int* predictions = KNN(dataset, 5);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
	*/
}

