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

void SSSP(const int numNodes, const int numEdges, int* d_inNeighbour, int* d_outNeighbour, long long int* d_weight, const int sourceVertex, long long int* d_dist, bool* d_relaxed, bool* d_flagA, bool* d_flagB);

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

    // pred[u] -> predecessor of vertex u in shortest path from sourceVertex ~~> u
    // dist[u] -> shortest distance of u from sourceVertex 
    long long int* d_dist;
    cudaMalloc(&d_dist, numNodes * sizeof(long long int));

    bool* d_negCycle;
    cudaMalloc(&d_negCycle, sizeof(bool));

    int* d_inNeighbour;
    cudaMalloc(&d_inNeighbour, numEdges * sizeof(int));
    cudaMemcpy(d_inNeighbour, inNeighbour, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    int* d_outNeighbour;
    cudaMalloc(&d_outNeighbour, numEdges * sizeof(int));
    cudaMemcpy(d_outNeighbour, outNeighbour, numEdges * sizeof(int), cudaMemcpyHostToDevice);

    long long* d_weight;
    cudaMalloc(&d_weight, numEdges * sizeof(long long int));
    cudaMemcpy(d_weight, weight, numEdges * sizeof(long long int), cudaMemcpyHostToDevice);

    bool* d_flagA;
    cudaMalloc(&d_flagA, numNodes * sizeof(bool));

    bool* d_flagB;
    cudaMalloc(&d_flagB, numNodes * sizeof(bool));

    std::vector <double> executionTimes(NUM_RUNS);

    double startTime, endTime;

    for(int run = 0; run < NUM_RUNS; run++){
        // Start time
        startTime = rtClock();

        SSSP(numNodes, numEdges, d_inNeighbour, d_outNeighbour, d_weight, sourceVertex, d_dist, d_negCycle, d_flagA, d_flagB);

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
    return;
}


// Grid stride loop
__global__ void initialize (const int numNodes, long long int* __restrict__ dist, bool* __restrict__ flagA, bool* __restrict__ flagB)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    while(__builtin_expect(u < numNodes, 1)){
        dist[u] = LLONG_MAX;
        flagA[u] = false;
        flagB[u] = false;
        u += (blockDim.x * gridDim.x);
    }
}

__global__ void initializeDistOfSourceVertex(long long int* __restrict__ dist, const int sourceVertex, bool* __restrict__ flagA)
{
    dist[sourceVertex] = 0;
    flagA[sourceVertex] = true;
}

// Grid stride again
__global__ void relaxEdges(const int numEdges, const int* __restrict__ inNeighbour, const int* __restrict__ outNeighbour, const long long int* __restrict__ weight, bool* relaxed, long long int* __restrict__ dist, bool* currFlag, bool* nextFlag)
{
    int e = threadIdx.x + blockIdx.x * blockDim.x;
    int u, v;
    long long int distCandidate, w;
    while(__builtin_expect(e < numEdges, 1)){
        u = inNeighbour[e];

        if(currFlag[u]){
            v = outNeighbour[e];
            w = weight[e];

            distCandidate = dist[u] + w;
            if(distCandidate < dist[v])
            {
                atomicMin(&dist[v], distCandidate);
                *relaxed = true;
                nextFlag[v] = true;
            }
        }
        e += (blockDim.x * gridDim.x);
     }
}

__global__ void unsetFlags (const int numNodes, bool* __restrict__ flag)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    while(__builtin_expect(u < numNodes, 1)){
        flag[u] = false;
        u += (blockDim.x * gridDim.x);
    }   
}

void SSSP(const int numNodes, const int numEdges, int* d_inNeighbour, int* d_outNeighbour, long long int* d_weight, const int sourceVertex, long long int* d_dist, bool* d_relaxed, bool* d_flagA, bool* d_flagB)
{
    // Initilializing distance and flags
    initialize <<< 256, 256 >>> (numNodes, d_dist, d_flagA, d_flagB);

    // Setting sourceVertex shortest distance to 0
    initializeDistOfSourceVertex <<< 1, 1 >>> (d_dist, sourceVertex, d_flagA); 

    bool relaxed = true;
    
    // Iterating atMost |V| - 1 times
    for(int iter = 1; (iter <= numNodes) && (relaxed); iter++)
    {
        cudaMemset(d_relaxed, false, sizeof(bool));

        if(iter & 1){
            relaxEdges <<< 256, 256 >>> (numEdges, d_inNeighbour, d_outNeighbour, d_weight, d_relaxed, d_dist, d_flagA, d_flagB);
            unsetFlags <<< 256, 256 >>> (numNodes, d_flagA);
        }else{
            relaxEdges <<< 256, 256 >>> (numEdges, d_inNeighbour, d_outNeighbour, d_weight, d_relaxed, d_dist, d_flagB, d_flagA);
            unsetFlags <<< 256, 256 >>> (numNodes, d_flagB);
        }
        
        cudaMemcpy(&relaxed, d_relaxed, sizeof(bool), cudaMemcpyDeviceToHost);
    }

    return;
}
