//============================================================================
// Name        : KNN_CUDA.cpp
// Author      : 
// Version     :
// Copyright   : 
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <limits>
#include <stdlib.h>
#include <stdint.h>
#include <float.h>
#include <math.h>
#include "libarff/arff_parser.h"
#include "libarff/arff_data.h"
using namespace std;



__global__ void KNN(float *dataset, int k, int instance_count, int attribute_count, float *k_distances, float *k_classes, long inf)
{

    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    float largest_array_distance = 0;
    int index_largest_distance;


    //for loop to look at other instances in array
    // j is the instance being compared to.
    for(int j = 0; j < instance_count; j++){

    	float distance = 0;

    	// CALCULATE DISTANCE BETWEEN J INSTANCE AND TID INSTANCE via ATTRIBUTES "D"
    	for(int d = 0; d < attribute_count - 1; d++){ // compute distance between two instances
    		float diff = dataset[tid * attribute_count + d] - dataset[j * attribute_count + d];
    		distance += diff * diff;
    	}
    	distance = sqrt(distance);

    	// PLACE DISTANCES AND CLASSES INTO ARRAYS FOR FIRST 5 - 6 INSTANCES DEPENDING ON CASE
    	if(tid < instance_count){
    		if(j < k){
    			k_classes[tid * k + j] = dataset[j * attribute_count + attribute_count - 1];
    			if (tid == j)
    				distance = inf; // set value to infinity because the instance shouldn't include itself in k
    			k_distances[tid * k + j] = distance;
    			if(distance > largest_array_distance){
    				largest_array_distance = distance;
    				index_largest_distance = j; // index is a value 0 - 4 in the case k = 5
    			}
    		}

    		if(j >= k){
    			if (distance < largest_array_distance && tid != j){
    				k_distances[tid * k + index_largest_distance] = distance; // replace the largest distance with the smaller one
    				k_classes[tid * k + index_largest_distance] = dataset[j * attribute_count + attribute_count - 1]; // set class of replaced
    				// Find new largest distance
    				float new_largest = 0;
    				float new_largest_index;
    				//look through the k values and find the largest distance
    				for(int i = tid * k; i < (tid+1) * k; i++){
    					float temp = k_distances[i];
    					if(temp > new_largest){
    						new_largest = temp;
    						new_largest_index = i % k; // put back in the range of 0-4 in the case of k = 5
    					}
    				}
    				index_largest_distance = new_largest_index;
    				largest_array_distance = new_largest;
    			}
    		}
    	}
   	}

}

__global__ void VoteOnClass(int instance_count, int class_count, int k, float * k_classes, int * classVotes, int * predictions){
	// VOTE ON CLASS by going through k_classes array (blocks of size k) and incrementing the index position in classValues array
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(tid < instance_count){
	    for(int i = tid * k; i < (tid+1) * k; i++){ // this goes through k_classes
	    	int the_class = k_classes[i];
	    	classVotes[tid * class_count + the_class] += 1;
	    }

	    // FIND CLASS WITH MOST VOTES
	    int max_votes = 0;
	    int max_votes_index = 0;
	    for(int i = tid * class_count; i < (tid+1) * class_count; i++){
	    	if(classVotes[i] > max_votes){
	    		max_votes = classVotes[i];
	    		max_votes_index = i % class_count;
	    	}
	    }

	    predictions[tid] = max_votes_index;

	}


}



int* computeConfusionMatrix(int* predictions, ArffData* dataset)
{
    int* confusionMatrix = (int*)calloc(dataset->num_classes() * dataset->num_classes(), sizeof(int)); // matriz size numberClasses x numberClasses

    for(int i = 0; i < dataset->num_instances(); i++) // for each instance compare the true class and predicted class
    {
        int trueClass = dataset->get_instance(i)->get(dataset->num_attributes() - 1)->operator int32();

        int predictedClass = predictions[i];

        confusionMatrix[trueClass*dataset->num_classes() + predictedClass]++;
    }

    return confusionMatrix;
}

float computeAccuracy(int* confusionMatrix, ArffData* dataset)
{
    int successfulPredictions = 0;

    for(int i = 0; i < dataset->num_classes(); i++)
    {
        successfulPredictions += confusionMatrix[i*dataset->num_classes() + i]; // elements in the diagonal are correct predictions
    }

    return successfulPredictions / (float) dataset->num_instances();
}



int main(int argc, char *argv[])
{
    if(argc != 3)
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
    int k = atoi(argv[2]);

    // Allocate other host memory
    int *h_predictions = (int *)malloc(instance_count * sizeof(int));
    float* h_Kdist =(float*)calloc(k * instance_count, sizeof(float));
    float* h_Kclasses=(float*)calloc(k * instance_count, sizeof(float));
    int* h_classVotes = (int*)calloc((attribute_count - 1) * instance_count, sizeof(int));

    // Allocate the device input vector A
    int *d_predictions;
    float *d_dataset;
    float *d_Kdist;
    float* d_Kclasses;
    int* d_classVotes;


    // infinity
    double inf = std::numeric_limits<double>::infinity();


    cudaMalloc(&d_predictions, instance_count * sizeof(int));
    cudaMalloc(&d_dataset, instance_count * attribute_count * sizeof(float));
    cudaMalloc(&d_Kdist, k * instance_count * sizeof(float));
    cudaMalloc(&d_Kclasses, k * instance_count * sizeof(float));
    cudaMalloc(&d_classVotes, (attribute_count - 1) * instance_count * sizeof(int));


    // Copy the host input vectors A and B in host memory to the device input vectors in
    cudaMemcpy(d_predictions, h_predictions, instance_count * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dataset, h_dataset, instance_count * attribute_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kdist, h_Kdist, k * instance_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Kclasses, h_Kclasses, k * instance_count * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_classVotes, h_classVotes, (attribute_count - 1) * instance_count * sizeof(int), cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (instance_count + threadsPerBlock - 1) / threadsPerBlock;

    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);

    // set k value (number of neighbors)
    KNN<<<blocksPerGrid, threadsPerBlock>>>(d_dataset, k, instance_count, attribute_count, d_Kdist, d_Kclasses, inf);
    VoteOnClass<<<blocksPerGrid, threadsPerBlock>>>(instance_count, class_count, k, d_Kclasses, d_classVotes, d_predictions);

    cudaMemcpy(h_predictions, d_predictions, instance_count * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Kclasses, d_Kclasses, k* instance_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_Kdist, d_Kdist, k* instance_count * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_classVotes, d_classVotes, (attribute_count - 1) * instance_count * sizeof(int), cudaMemcpyDeviceToHost);

    int* confusionMatrix = computeConfusionMatrix(h_predictions, dataset);
    float accuracy = computeAccuracy(confusionMatrix, dataset);

    clock_gettime(CLOCK_MONOTONIC_RAW, &end);
    uint64_t diff = (1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec) / 1e6;

    printf("The KNN classifier for %lu instances required %llu ms CPU time, accuracy was %.4f\n", dataset->num_instances(), (long long unsigned int) diff, accuracy);
}

