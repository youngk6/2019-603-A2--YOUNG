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

__global__ void KNN(int *predictions, float *dataset, int k, int instance_count, int attribute_count, float *k_distances, float *k_classes, int class_count)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float largest_array_distance = 0;
    int index_largest_distance;

    //for loop to look at other instances in array
    // j is the instance being compared to.
    int comp_cnt = 0;
    for(int j = 0; j < instance_count; j++){
    	if(j == tid) continue;

    	float distance = 0;

    	// CALCULATE DISTANCE BETWEEN J INSTANCE AND TID INSTANCE via ATTRIBUTES "D"
    	for(int d = 0; d < attribute_count - 1; d++){ // compute distance between two instances
    		float diff = dataset[tid * attribute_count + d] - dataset[j * attribute_count + d];
    		distance += diff * diff;
    	}
    	distance = sqrt(distance);

    	// PLACE DISTANCES AND CLASSES INTO ARRAYS FOR FIRST 5 - 6 INSTANCES DEPENDING ON CASE
    	if(j <= k){
    		if(tid <= k && tid != k){					// FILLS THE CLASSES ARRAY WITH FIRST FIVE THAT ARENT ITSELF
    			k_distances[tid * k + comp_cnt] = distance;
    			k_classes[tid * k + comp_cnt] = dataset[tid * attribute_count + attribute_count - 1];
    			if(distance > largest_array_distance){
    				largest_array_distance = distance;
    				index_largest_distance = comp_cnt;
    			}
    			comp_cnt++;
    		}else{
    			k_distances[tid * k + j] = distance;
    			k_classes[tid * k + j] = dataset[tid * attribute_count + attribute_count - 1];
    			if(distance > largest_array_distance){
    			    largest_array_distance = distance;
    			    index_largest_distance = j;
    			 }
    		}
    	}

    }

}

int main(int argc, char *argv[])
{
    if(argc != 2)
    {
        cout << "Usage: ./main datasets/datasetFile.arff" << endl;
        exit(0);
    }

    // READ IN DATASET
    ArffParser parser(argv[1]);
    ArffData *dataset = parser.parse();

    // GET METRICS FROM DATASET
    int instance_count = dataset->num_instances();
    int attribute_count = dataset->num_attributes();
    int class_count = dataset->num_classes();

    // MAKE DATASET INTO 1D ARRAY
    float *h_dataset = (float *)malloc(instance_count * attribute_count * sizeof(float));
    int count = 0;
    for (int i = 0; i < instance_count; i++){
    	for (int j = 0; j < attribute_count; j++){
    		h_dataset[count] = dataset->get_instance(i)->get(j)->operator float();
    		count++;
    	}
    }

    // START CLOCK
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC_RAW, &start);

    // SET K
    int k = 5;

    // Allocate other host memory
    int *h_predictions = (int *)malloc(instance_count * sizeof(int));
    float* h_Kdist =(float*)calloc(k * instance_count, sizeof(float));
    float* h_Kclasses=(float*)calloc(k * instance_count, sizeof(float));

    // Allocate the device input vector A
    int *d_predictions;
    float *d_distance_calculations; /// maybe get rid of?
    int *d_arr_class_data;
    float *d_arr_data;
    float *d_Kdist;
    float* d_Kclasses;


    cudaMalloc(&d_predictions, instance_count * sizeof(int));
    cudaMalloc(&d_distance_calculations, instance_count *sizeof(float));
    cudaMalloc(&d_arr_class_data, instance_count * sizeof(int));
    cudaMalloc(&d_arr_data, instance_count * attribute_count * sizeof(float));
    cudaMalloc(&d_Kdist, k * instance_count * sizeof(float));
    cudaMalloc(&d_Kclasses, k * instance_count * sizeof(float));

    // Copy the host input vectors A and B in host memory to the device input vectors in
    cudaMemcpy(d_predictions, h_predictions, instance_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_data, h_dataset, instance_count * attribute_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kdist, h_Kdist, k * instance_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kclasses, h_Kclasses, k * instance_count * sizeof(float), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (instance_count + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // set k value (number of neighbors)
    KNN<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_arr_data, k, instance_count, attribute_count, d_Kdist, d_Kclasses,class_count);

    cudaMemcpy(h_predictions, d_predictions, instance_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Kclasses, d_Kclasses, k* instance_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Kdist, d_Kdist, k* instance_count * sizeof(float), cudaMemcpyDeviceToHost);

    for(int i = 0; i < 336 * 5; i++){
    	cout << h_Kdist[i];
    }

    /*

    int* predictions = KNN(dataset, 5);
    int* confusionMatrix = computeConfusionMatrix(predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
	*/
}

