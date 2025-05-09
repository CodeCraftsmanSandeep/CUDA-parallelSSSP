#include <stdio.h>
#include <cuda.h>
#include <vector>

double rtClock();

#ifndef Edge_D
#define Edge_D

struct Edge{
    int u, v;
    long long int w;
};

#endif 

double findMean(const std::vector <double> times);

double findMedian(std::vector <double> times);

double findStandardDeviation(const std::vector<double> times);

constexpr unsigned int NUM_RUNS = 11;

template <typename T>
void print(T* arr, int size)
{
    for(int i = 0; i < size; i++)
    {
        printf("%d ", arr[i]);
    }
    printf("\n");
  }

void SSSP(
      const int numNodes,
      const int numEdges,
      const int* __restrict__ d_rowPtr,
      const int* __restrict__ d_colInd,
      const long long int* __restrict__ d_weights,
      const int sourceVertex,
      long long int* __restrict__ d_dist,
      bool* __restrict__ d_relaxed,
      bool* __restrict__ d_flagA,
      bool* __restrict__ d_flagB);

void solve(int numNodes, int numEdges, Edge* edgeList, int sourceVertex) 
{
    // Creating CSR format from edgeList
    int* rowPtr = (int*)calloc(numNodes + 1, sizeof(int));
    int* colInd = (int*)malloc(numEdges * sizeof(int));
    long long int* weights = (long long int*)malloc(numEdges * sizeof(long long int));

    for(int edge = 0; edge < numEdges; edge++)
    {
        rowPtr[edgeList[edge].u+1]++;
    }

    for(int u = 0; u < numNodes; u++)
    {
        rowPtr[u+1] += rowPtr[u];
    }

    int* currentPos = (int*)calloc(numNodes, sizeof(int)); // Auxillary array to track insertion offset
    for(int edge = 0; edge < numEdges; edge++)
    {
        colInd[rowPtr[edgeList[edge].u] + currentPos[edgeList[edge].u]] = edgeList[edge].v;
        weights[rowPtr[edgeList[edge].u] + currentPos[edgeList[edge].u]] = edgeList[edge].w;
        currentPos[edgeList[edge].u]++;
    }

    // print <int> (rowPtr, numNodes + 1);
    // print <int> (colInd, numEdges);
    // print <long long int> (weights, numEdges);

    // Copying graph in cpu to gpu (CSR format)
    int* d_rowPtr;
    cudaMalloc(&d_rowPtr, (numNodes + 1) * sizeof(int));
    cudaMemcpy(d_rowPtr, rowPtr, sizeof(int) * (numNodes + 1), cudaMemcpyHostToDevice);

    int* d_colInd;
    cudaMalloc(&d_colInd, numEdges * sizeof(int));
    cudaMemcpy(d_colInd, colInd, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    long long int* d_weights;
    cudaMalloc(&d_weights, numEdges * sizeof(long long int));
    cudaMemcpy(d_weights, weights, numEdges * sizeof(long long int), cudaMemcpyHostToDevice);

    long long int* d_dist; // dist[u] -> shortest distance of u from sourceVertex 
    cudaMalloc(&d_dist, numNodes * sizeof(long long int));

    bool* d_negCycle;
    cudaMalloc(&d_negCycle, sizeof(bool));

    bool* d_flagA;
    cudaMalloc(&d_flagA, numNodes * sizeof(bool));

    bool* d_flagB;
    cudaMalloc(&d_flagB, numNodes * sizeof(bool));

    std::vector <double> executionTimes(NUM_RUNS);

    double startTime, endTime;

    for(int run = 0; run < NUM_RUNS; run++){
        // Start time
        startTime = rtClock();

        SSSP(numNodes, numEdges, d_rowPtr, d_colInd, d_weights, sourceVertex, d_dist, d_negCycle, d_flagA, d_flagB);

        // End time
        endTime = rtClock();

        executionTimes[run] = (endTime - startTime) * 1e3; // in milli-seconds
    }

    printf("Num-runs,Mean-time(ms),Median-time(ms),Std-deviation(ms)\n");
    printf("%d,%.6f,%.6f,%.6f\n", NUM_RUNS, findMean(executionTimes), findMedian(executionTimes), findStandardDeviation(executionTimes));

    long long int* dist = (long long int*)malloc(numNodes * sizeof(long long int));
    cudaMemcpy(dist, d_dist, numNodes * sizeof(long long int), cudaMemcpyDeviceToHost);

    bool* negCycle = (bool*)malloc(sizeof(bool));
    cudaMemcpy(negCycle, d_negCycle, sizeof(bool), cudaMemcpyDeviceToHost);
    
    if(*negCycle){
        printf("Negative Cycle Detected!\n");
    }else{
        printf("Vertex,Distance\n");
        for(int u = 0; u < numNodes; u++){
            if(dist[u] == LLONG_MAX){
                printf("%d,INF\n", u);
            }else{
                printf("%d,%lld\n", u, dist[u]);
            }
        }
    }

    free(negCycle);
    free(dist);

    cudaFree(d_rowPtr);
    cudaFree(d_colInd);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_negCycle);
    cudaFree(d_flagA);
    cudaFree(d_flagB);

    return;
}


// Setting distances of vertices as infinity, and all flags unset
__global__ void initialize(
        const int numNodes, 
        long long int* __restrict__ dist, 
        bool* __restrict__ currFlag, 
        bool* __restrict__ nextFlag)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    while(__builtin_expect(u < numNodes, 1)){
        dist[u] = LLONG_MAX;
        currFlag[u] = nextFlag[u] = false;
        u += (blockDim.x * gridDim.x);
    }
}

// Setting distance of source vertex as 0
__global__ void initializeDistOfSourceVertex(
                        long long int* __restrict__ dist, 
                        const int sourceVertex, 
                        bool* __restrict__ currFlag)
{
    dist[sourceVertex] = 0;
    currFlag[sourceVertex] = true;
}

// ensure that blockDim.x = 32
// relaxed out edges of vertices whoose distance is reduced in the previous iteration
__global__ void relaxEdges(
        const int            numNodes,
        const int*           __restrict__ rowPtr, 
        const int*           __restrict__ colInd, 
        const long long int* __restrict__ weights, 
        long long int*       __restrict__ dist, 
        bool*                __restrict__ currFlag,     
        bool*                __restrict__ nextFlag,
        bool*                __restrict__ relaxed)
{
    int u = threadIdx.y + blockIdx.x * blockDim.y;
    int v;
    long long int distCandidate, w;
    while(__builtin_expect(u < numNodes, 1))
    {
        if(currFlag[u])
        {
            __syncwarp();
            currFlag[u] = false;

            int outEdge = rowPtr[u] + threadIdx.x;
            while(outEdge < rowPtr[u+1])
            {
                v = colInd[outEdge];
                w = weights[outEdge];

                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {
                    atomicMin(&dist[v], distCandidate); // L2 cache is coherent for atomics, so no cache coherence problem
                    nextFlag[v] = true;
                    *relaxed = true;
                }

                outEdge += blockDim.x;
            }
        }
        u += blockDim.y * gridDim.x;
    }
}

void SSSP(  
      const int numNodes, 
      const int numEdges, 
      const int* __restrict__ d_rowPtr,
      const int* __restrict__ d_colInd, 
      const long long int* __restrict__ d_weights, 
      const int sourceVertex, 
      long long int* __restrict__ d_dist, 
      bool* __restrict__ d_relaxed, 
      bool* __restrict__ d_flagA, 
      bool* __restrict__ d_flagB) 
{
    // Initilializing distance and flags
    initialize <<< 256, 256 >>> (numNodes, d_dist, d_flagA, d_flagB);

    // Setting sourceVertex shortest distance to 0
    initializeDistOfSourceVertex <<< 1, 1 >>> (d_dist, sourceVertex, d_flagA); 

    bool relaxed = true;

    dim3 blockSize(32, 16); // 16 warps
    dim3 gridSize(256, 1);

    // Iterating atMost |V| times (the |V|th iteration if so happens will be executed to detect negative cycle)
    for(int iter = 1; (iter <= numNodes) && relaxed; iter++)
    {
        cudaMemset(d_relaxed, false, sizeof(bool));

        if(iter & 1){
            relaxEdges <<< gridSize, blockSize >>> (numNodes, d_rowPtr, d_colInd, d_weights, d_dist, d_flagA, d_flagB, d_relaxed);
        }else{
            relaxEdges <<< gridSize, blockSize >>> (numNodes, d_rowPtr, d_colInd, d_weights, d_dist, d_flagB, d_flagA, d_relaxed);
        }
        cudaMemcpy(&relaxed, d_relaxed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    return;
}
