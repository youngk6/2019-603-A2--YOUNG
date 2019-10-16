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

__global__ void KNN(int *predictions, float *dataset, int *classes, float *distance_calculations, int k_neighbors, int num_elements, int num_attributes, float *Arr_dist, int *Arr_classes)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float largest_array_distance = 0;
    int index_largest_distance;

    //for loop to look at other instances in array
    for(int j = 0; j < num_elements; j++){
    	if(j == tid) continue;

    	float distance = 0;
    	// dataset has 7 attributes per instance
    	// each instance is in a class
    	// to get beginning of a class, 7 * tid

    	for(int k = 0; k < num_attributes; k++){ // compute distance between two instances
    		// compare thread attribute k with other thread attribute k
    		float diff = dataset[tid * num_attributes + k] - dataset[j * num_attributes + k];
    		distance += diff * diff;
    	}
    	distance = sqrt(distance);


	    // THIS IS INITIALIZING ARRAY AUTOMATICALLY WITH FIRST K VALUES. Get a starting point.
	    if(j < k_neighbors){
                Arr_dist[j] = distance; //put distance in
                Arr_classes[j] = classes[j]; // put class in
                if(distance > largest_array_distance){ // keep track of largest distance in array
                	largest_array_distance = distance;
                	index_largest_distance = j;
                }
	    }

	    if(j >= k_neighbors){
	    		if(distance < largest_array_distance){ // IF THERE IS A CLOSER NEIGHBOR THAT SHOULD BE IN ARRAY
	    		    Arr_dist[index_largest_distance] = distance; // change the distance, then add the class
	    		    Arr_classes[index_largest_distance] = classes[j];
	    		    //FIND NEW LARGEST DISTANCE
	    		    float new_largest = 0;
	    		    int new_largest_index;
	    		    for(int r = 0; r < k_neighbors; r++){
	    		    	float temp = Arr_dist[r];
	    		    	if(temp > new_largest){
	    		    		new_largest = temp;
	    		    		new_largest_index = r;
	    		    	}
	     		    }
	    		    largest_array_distance = new_largest;
	    		    index_largest_distance  = new_largest_index;
	    		}
	    }

	    //next vote on class and update predictions.


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

    // set k
    int k = 5;

    // Allocate other host memory
    int *h_predictions = (int *)malloc(num_elements * sizeof(int));
    float * h_distance_calculations = (float *)calloc(num_elements, sizeof(float));
    float* h_Arr_dist =(float*)calloc(k, sizeof(float));
    int* h_Arr_classes=(int*)calloc(k, sizeof(int));

    // Allocate the device input vector A
    int *d_predictions;
    float *d_distance_calculations; /// maybe get rid of?
    int *d_arr_class_data;
    float *d_arr_data;
    float *d_Arr_dist;
    int* d_Arr_classes;


    cudaMalloc(&d_predictions, num_elements * sizeof(int));
    cudaMalloc(&d_distance_calculations, num_elements *sizeof(float));
    cudaMalloc(&d_arr_class_data, num_elements * sizeof(int));
    cudaMalloc(&d_arr_data, num_elements * total_attributes * sizeof(float));
    cudaMalloc(&d_Arr_dist, k * sizeof(float));
    cudaMalloc(&d_Arr_classes, k * sizeof(int));

    // Copy the host input vectors A and B in host memory to the device input vectors in
    cudaMemcpy(d_predictions, h_predictions, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_distance_calculations, h_distance_calculations, num_elements *sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_class_data, h_arr_class_data, num_elements * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_arr_data, h_arr_data, num_elements * total_attributes * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Arr_dist, h_Arr_dist, k * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Arr_classes, h_Arr_classes, k * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // set k value (number of neighbors)
    KNN<<<blocksPerGrid, threadsPerBlock>>>(d_predictions, d_arr_data, d_arr_class_data, d_distance_calculations, k, num_elements, total_attributes, d_Arr_dist, d_Arr_classes);

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

