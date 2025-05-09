#include <stdio.h>
#include <cuda.h>
#include <vector>

#define cudaCheckError(code)                                               \
{                                                                          \
    if ((code) != cudaSuccess) {                                           \
        fprintf(stderr, "Cuda failure %s:%d: '%s' \n", __FILE__, __LINE__, \
        cudaGetErrorString(code));                                         \
    }                                                                      \
}

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
int numIterations = 0;

void SSSP(const int numNodes, const int numEdges, int* d_inNeighbour, int* d_outNeighbour, long long int* d_weight, const int sourceVertex, long long int* d_dist, bool* d_relaxed);
void solve(int numNodes, int numEdges, Edge* edgeList, int sourceVertex) 
{ 
    // Converting AoS to SoA
    int* inNeighbour = (int*)malloc(numEdges * sizeof(int));
    int* outNeighbour = (int*)malloc(numEdges * sizeof(int));
    long long int* weight = (long long int*)malloc(numEdges * sizeof(long long int));
    for(int e = 0; e < numEdges; e++){
        inNeighbour[e]  = edgeList[e].u;
        outNeighbour[e] = edgeList[e].v;
        weight[e]       = edgeList[e].w;
    }

    // dist[u] -> shortest distance of u from sourceVertex 
    long long int* d_dist;
    cudaCheckError(cudaMalloc(&d_dist, numNodes * sizeof(long long int)));

    bool* d_negCycle;
    cudaCheckError(cudaMalloc(&d_negCycle, sizeof(bool)));

    int* d_inNeighbour;
    cudaCheckError(cudaMalloc(&d_inNeighbour, numEdges * sizeof(int)));
    cudaMemcpy(d_inNeighbour, inNeighbour, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    int* d_outNeighbour;
    cudaCheckError(cudaMalloc(&d_outNeighbour, numEdges * sizeof(int)));
    cudaMemcpy(d_outNeighbour, outNeighbour, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    long long* d_weight;
    cudaCheckError(cudaMalloc(&d_weight, numEdges * sizeof(long long int)));
    cudaMemcpy(d_weight, weight, numEdges * sizeof(long long int), cudaMemcpyHostToDevice);

    std::vector <double> executionTimes(NUM_RUNS);

    double startTime, endTime;

    for(int run = 0; run < NUM_RUNS; run++){
        // Start time
        numIterations = 0;
        startTime = rtClock();

        SSSP(numNodes, numEdges, d_inNeighbour, d_outNeighbour, d_weight, sourceVertex, d_dist, d_negCycle);

        // End time
        endTime = rtClock();

        executionTimes[run] = (endTime - startTime) * 1e3; // in milli-seconds
    }

    printf("Num-runs,Mean-time(ms),Median-time(ms),Std-deviation(ms),Num-iterations\n");
    printf("%d,%.6f,%.6f,%.6f,%d\n", NUM_RUNS, findMean(executionTimes), findMedian(executionTimes), findStandardDeviation(executionTimes), numIterations);

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
    return;
}

// Grid stride loop
__global__ void initialize (const int numNodes, long long int* __restrict__ dist)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    while(__builtin_expect(u < numNodes, 1)){
        dist[u] = LLONG_MAX;
        u += (blockDim.x * gridDim.x);
    }
}

__global__ void initializeDistOfSourceVertex(long long int* __restrict__ dist, const int sourceVertex)
{
    dist[sourceVertex] = 0;
}

// Grid stride access
__global__ void relaxEdges(const unsigned int                 numEdges, 
                           const int* __restrict__            inNeighbour, 
                           const int* __restrict__            outNeighbour, 
                           const long long int* __restrict__  weight, 
                           bool* __restrict__                 relaxed, 
                           long long int* __restrict__        dist)
{
    unsigned int e = threadIdx.x + blockIdx.x * blockDim.x;
    int u, v;
    long long int distCandidate;
    while(__builtin_expect(e < numEdges, 1)){
        u = inNeighbour[e];

        if(dist[u] != LLONG_MAX){
            v = outNeighbour[e];

            distCandidate = dist[u] + weight[e];;
            if(distCandidate < dist[v])
            {
                atomicMin(&dist[v], distCandidate);
                *relaxed = true;
            }
        }
        e += (blockDim.x * gridDim.x);
     }
}

void SSSP(const int numNodes, const int numEdges, int* d_inNeighbour, int* d_outNeighbour, long long int* d_weight, const int sourceVertex, long long int* d_dist, bool* d_relaxed)
{
    // Initilializing predecessors and distance
    initialize <<< 256, 256 >>> (numNodes, d_dist);

    // Setting sourceVertex shortest distance to 0
    initializeDistOfSourceVertex <<< 1, 1 >>> (d_dist, sourceVertex); 

    bool relaxed = true;
    
    // Iterating atMost |V| - 1 times
    for(int iter = 1; (iter <= numNodes) && relaxed; iter++)
    {
        // numIterations++;
        cudaMemset(d_relaxed, false, sizeof(bool));
        relaxEdges <<< 256, 256 >>> (numEdges, d_inNeighbour, d_outNeighbour, d_weight, d_relaxed, d_dist);
        cudaMemcpy(&relaxed, d_relaxed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    return;
}
