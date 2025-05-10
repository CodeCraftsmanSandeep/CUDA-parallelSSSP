#include <stdio.h>
#include <cuda.h>
#include <vector>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/tuple.h>
#include <thrust/iterator/zip_iterator.h>

double rtClock();

#ifndef Edge_D
#define Edge_D
struct Edge{
    int u, v;
    long long int w;
};
#endif 

#ifndef Queries_D
#define Queries_D
struct Queries
{
    char command;
    int n;
    union
    {
        int* verticesToKnow;
        Edge* edgesToAdd;
    } data;
};
#endif

struct EdgeNode 
{
    int v;
    long long int w;

    EdgeNode (int v, int w): v(v), w(w) {}
};

double findMean(const std::vector <double> times);

double findMedian(std::vector <double> times);

double findStandardDeviation(const std::vector<double> times);

constexpr unsigned int NUM_RUNS = 11;

constexpr unsigned int BLOCK_SIZE = 256;
constexpr unsigned int GRID_SIZE = 256;

/* Kernel
    ~> To allocate few bytes of memory to each adjacency list of each vertex */
__global__ void initializeAdjList(
        const int                    numNodes,
        int** __restrict__           adjListDestinationVertices,
        long long int** __restrict__ adjListWeights,
        int* __restrict__            adjListCapacities,
        int* __restrict__            adjListSizes,
        const int                    initialCapacity)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;

    while(u < numNodes)
    {
        adjListDestinationVertices[u] = (int*)malloc(initialCapacity * sizeof(int));
        adjListWeights[u] = (long long int*)malloc(initialCapacity * sizeof(long long int));
        adjListCapacities[u] = initialCapacity;
        adjListSizes[u] = 0;

        u += gridDim.x * blockDim.x;
    }
}

/* Kernel
    ~> To erase memory alloted to adjacency list of each vertex */
__global__ void eraseAdjList(
        const int                    numNodes,
        int** __restrict__           adjListDestinationVertices,
        long long int** __restrict__ adjListWeights)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    while(u < numNodes)
    {
        free(adjListDestinationVertices[u]);
        free(adjListWeights[u]);

        u += gridDim.x * blockDim.x;
    }
}

class GraphAdjList
{
protected:
    int numNodes;

    // SoA (Structure of Arrays) representation of adjacency list for efficient memory access patterns in cuda
    // Tuple (destinationVertices[u][i], weights[u][i]) is ith edge u~>destinationVertices[u][i] with edge weight weights[u][i]
    int**           d_adjListDestinationVertices;
    long long int** d_adjListWeights;

    // Auxillary data structures to support adjacency list
    int*            d_adjListCapacities;
    int*            d_adjListSizes;
    const int       initialCapacity = 32;

public:
    // Constructor for: Graph with no edges
    GraphAdjList(int numNodes): numNodes(numNodes)
    {
        cudaMalloc(&d_adjListDestinationVertices, numNodes * sizeof(int*));
        cudaMalloc(&d_adjListWeights, numNodes * sizeof(long long int*));
        cudaMalloc(&d_adjListCapacities, numNodes * sizeof(int));
        cudaMalloc(&d_adjListSizes, numNodes * sizeof(int));
        
        cudaDeviceSetLimit(cudaLimitMallocHeapSize, (1e9 * 1LL * sizeof(int)));
        initializeAdjList <<< GRID_SIZE, BLOCK_SIZE >>> (numNodes, d_adjListDestinationVertices, d_adjListWeights, d_adjListCapacities, d_adjListSizes, initialCapacity);
        cudaDeviceSynchronize();
    }

    // Destructor for adjacency list
    ~GraphAdjList()
    {
        // Erase adjlist allocated to each vertex
        eraseAdjList <<< GRID_SIZE, BLOCK_SIZE >>> (numNodes, d_adjListDestinationVertices, d_adjListWeights);
        cudaDeviceSynchronize();

        // Erasing pointers to adjlist of each vertex
        cudaFree(d_adjListDestinationVertices);
        cudaFree(d_adjListWeights);

        // Erasing memory allocated to auxillary data structures
        cudaFree(d_adjListCapacities);
        cudaFree(d_adjListSizes);
    }
};

/* Kernel to
    ~> Insert batch of edges into existing graph */
__global__ void insertBatchOfEdgesKernel(
        const int                           numNodes,
        const int* __restrict__             batchDestinationVertices,
        const long long int* __restrict__   batchWeights,
        const int* __restrict__             batchPos,
        int* __restrict__                   adjListSizes,
        int* __restrict__                   adjListCapacities,
        int** __restrict__                  adjListDestinationVertices,
        long long int** __restrict__        adjListWeights,
        bool* __restrict__                  relaxed,
        bool* __restrict__                  nextFlag, 
        long long int* __restrict__         dist)
{
    int u = threadIdx.y + blockIdx.x * blockDim.y;
    int v;
    long long int distCandidate, w;
    while(__builtin_expect(u < numNodes, 1))
    {
        const int start     = batchPos[u];
        const int end       = batchPos[u+1];
        const int reqSpace  = end-start;
        const int oldSize   = adjListSizes[u];
        const int newSize   = oldSize + reqSpace;

        if(newSize > adjListCapacities[u])
        {
            int* newStorageForDestinationVertices;
            long long int* newStorageForWeights;
            const int newCapacity = (newSize << 1);

            if(threadIdx.x == 0)
            {
                // insert space in the adjacency list of vertex
                newStorageForDestinationVertices = (int*)malloc(newCapacity * sizeof(int));
                newStorageForWeights = (long long int*)malloc(newCapacity * sizeof(long long int));
            }
            
            // No __syncwarp() Needed because: __shfl_sync inherently synchronizes warp
            // Broadcast the pointer from lane 0 to all lanes in the warp
            newStorageForDestinationVertices = (int*)__shfl_sync(0xFFFFFFFF, (uintptr_t)newStorageForDestinationVertices, 0);
            newStorageForWeights = (long long int*)__shfl_sync(0xFFFFFFFF, (uintptr_t)newStorageForWeights, 0);

            if(newStorageForDestinationVertices == nullptr || newStorageForWeights == nullptr)
            {
                printf("Memory allocation failed\n");
                continue;
            }
            // Copy the previosuly present out neighbours to new storage
            int iter = threadIdx.x;
            while(iter < adjListSizes[u])
            {
                newStorageForDestinationVertices[iter] = adjListDestinationVertices[u][iter];
                newStorageForWeights[iter] = adjListWeights[u][iter];

                iter += 32;
            }
            __syncwarp();  // Needed because: ensuring copying is completed by threads in warp before freeing the memory

            if(threadIdx.x == 0)
            {
                free(adjListDestinationVertices[u]);
                free(adjListWeights[u]);

                adjListDestinationVertices[u] = newStorageForDestinationVertices;
                adjListWeights[u] = newStorageForWeights;
                
                adjListSizes[u] = newSize;
                adjListCapacities[u] = newCapacity;
            }
            __syncwarp();  // Needed because: ensuring new pointers are copied to adjList of vertex u before adding new edges
        }
        else if(threadIdx.x == 0)
        {
            adjListSizes[u] = newSize;
        }

        int adjListIter = oldSize + threadIdx.x;
        int batchIter = start + threadIdx.x;
        while(batchIter < end)
        {
            v = adjListDestinationVertices[u][adjListIter] = batchDestinationVertices[batchIter];
            w = adjListWeights[u][adjListIter] = batchWeights[batchIter];

            if(dist[u] != LLONG_MAX){
                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {
                    atomicMin(&dist[v], distCandidate);
                    *relaxed = true;
                    nextFlag[v] = true;
                }
            }
            // warp-size: which is blockDim.x
            batchIter   += 32; 
            adjListIter += 32; 
        }

        u += blockDim.y * gridDim.x;
    }
}

/* Kernel
    ~> which relaxes edges in parallel */
__global__ void relaxEdges(
        const int                           numNodes,
        const int* __restrict__             adjListSizes,
        int** __restrict__            adjListDestinationVertices,
        long long int** __restrict__  adjListWeights,
        bool* __restrict__                  currFlag,
        bool* __restrict__                  nextFlag,
        long long int* __restrict__         dist,
        bool* __restrict__                  relaxed)
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

            int outEdge = threadIdx.x;
            while(outEdge < adjListSizes[u])
            {
                v = adjListDestinationVertices[u][outEdge];
                w = adjListWeights[u][outEdge];

                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {
                    atomicMin(&dist[v], distCandidate); // L2 cache is coherent for atomics, so no cache coherence problem
                    nextFlag[v] = true;
                    *relaxed = true;
                }
                outEdge += 32; // warp-size which is blockDim.x here
            }
        }
        u += blockDim.y * gridDim.x;
    }
}

/* Kernel to
   ~> set distance of every vectex LLONG_MAX */
__global__ void initializeDistanceValues(
        const int                   numNodes,
        long long int* __restrict__ dist)
{
    int u = threadIdx.x + blockIdx.x * blockDim.x;
    while(__builtin_expect(u < numNodes, 1))
    {
        dist[u] = LLONG_MAX;
        u += blockDim.x * gridDim.x;
    }
}

/* Kernel to
    ~> set distance of source vertex to 0
    ~> negative cycle to false */
__global__ void initializeGraphSitutaion(
        bool* __restrict__          negCycle,
        long long int* __restrict__ dist,
        const int                   sourceVertex)
{
    *negCycle = false;
    dist[sourceVertex] = 0;
}

/* Kernel to
    ~> to find start and end positions to insert edges in batch */
__global__ void computeIndices(
        int* __restrict__ batchOutDegree,
        const int* __restrict__ sourceVertices,
        const int batchSize)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    while(__builtin_expect(index < batchSize, 1))
    {
        atomicInc((unsigned int*)&batchOutDegree[sourceVertices[index] + 1], UINT_MAX);
        index += blockDim.x * gridDim.x;
    }
}

/*
GraphSSSP class
    -> Private inheritance from GraphAdjList class. (private because adjList is not exposed to outside class)
    -> Private inheritance from 
    -> Provides API:
        i) Batch insert
        ii) Batch query (get shortestdistances of batch of vertices)
*/
class GraphSSSP : private GraphAdjList
{
    long long int* d_dist;
    long long int* dist;
    bool*          d_negCycle;
    bool*          negCycle;
    int            sourceVertex;

    // Auxillary data structures and storage
    int*           d_batchOutDegree;
    int*           d_tempStorage;
    const size_t   tempStorageBytes = 1e5 * sizeof(int);
    bool*          d_relaxed;
    bool*          relaxed;
    bool*          d_flagA;
    bool*          d_flagB;
    double         startTime, endTime, timeConsumed;

    void runPartialSSSP()
    {
        bool relaxed = true;
        dim3 BLOCK_SIZE(32, 16);
        for(int iter = 1; (iter <= numNodes) && (relaxed); iter++)
        {
            relaxed = false;
            cudaMemset(d_relaxed, false, sizeof(bool));

            if(iter & 1) relaxEdges <<< GRID_SIZE, BLOCK_SIZE >>> (numNodes, d_adjListSizes, d_adjListDestinationVertices, d_adjListWeights, d_flagA, d_flagB, d_dist, d_relaxed);
            else         relaxEdges <<< GRID_SIZE, BLOCK_SIZE >>> (numNodes, d_adjListSizes, d_adjListDestinationVertices, d_adjListWeights, d_flagB, d_flagA, d_dist, d_relaxed);
            
            cudaMemcpy(&relaxed, d_relaxed, sizeof(bool), cudaMemcpyDeviceToHost);
        }
        *negCycle = relaxed;
        cudaMemcpy(d_negCycle, negCycle, sizeof(bool), cudaMemcpyHostToDevice);
    }
public:
    // Graph with no edges
    GraphSSSP(int numNodes, int sourceVertex): GraphAdjList(numNodes), sourceVertex(sourceVertex)
    {
        // Allocating memory for shortest distance values.
        cudaMalloc(&d_dist, numNodes * sizeof(long long int)); // On gpu
        dist = (long long int*)malloc(numNodes * sizeof(long long int)); // On cpu

        // Allocating memory for negative cycle flag
        cudaMalloc(&d_negCycle, sizeof(bool));
        negCycle = (bool*)malloc(sizeof(bool));

        // Initialize distance values on gpu
        initializeDistanceValues <<< GRID_SIZE, BLOCK_SIZE >>> (numNodes, d_dist);

        // No negative cycle
        initializeGraphSitutaion <<< 1, 1 >>> (d_negCycle, d_dist, sourceVertex);
        
        // Copy from device to host
        cudaMemcpy(dist, d_dist, numNodes * sizeof(long long int), cudaMemcpyDeviceToHost);
        cudaMemcpy(negCycle, d_negCycle, sizeof(bool), cudaMemcpyDeviceToHost);

        // Allocating memory for auxillary data structures
        cudaMalloc(&d_batchOutDegree, (numNodes + 1) * sizeof(int));
        cudaMalloc(&d_tempStorage, tempStorageBytes); // allocating temporary storage which can be used in libraries
        cudaMalloc(&d_relaxed, sizeof(bool));
        cudaMalloc(&d_flagA, numNodes * sizeof(bool));
        cudaMalloc(&d_flagB, numNodes * sizeof(bool));

        // Initializing auxillary data structures
        cudaMemset(d_flagA, false, numNodes * sizeof(bool));
        cudaMemset(d_flagB, false, numNodes * sizeof(bool));
    }

    ~GraphSSSP()
    {
        // Deallocating memory of shortest-distance values
        cudaFree(d_dist); // On gpu
        free(dist); // On cpu

        // Deallocating memory of negative cycle flag
        cudaFree(d_negCycle); //  On gpu
        free(negCycle); // On cpu

        // Deallocating memory of auxillary data structures
        cudaFree(d_batchOutDegree);

        cudaFree(d_relaxed);
        cudaFree(d_flagA);
        cudaFree(d_flagB);
    }
   
    void batchInsert(int batchSize, Edge* edgeList)
    {
        // Allocating memory for conversion from AoS to SoA type (for edges)
        int* d_sourceVertices;
        int* sourceVertices;
        cudaMalloc(&d_sourceVertices, batchSize * sizeof(int));
        sourceVertices = (int*)malloc(batchSize * sizeof(int));

        int* d_destinationVertices;
        int* destinationVertices;
        cudaMalloc(&d_destinationVertices, batchSize * sizeof(int));
        destinationVertices = (int*)malloc(batchSize * sizeof(int));

        long long int* d_weights;
        long long int* weights;
        cudaMalloc(&d_weights, batchSize * sizeof(long long int));
        weights = (long long int*)malloc(batchSize * sizeof(long long int));

        for(int e = 0; e < batchSize; e++)
        {
            sourceVertices[e] = edgeList[e].u;
            destinationVertices[e] = edgeList[e].v;
            weights[e] = edgeList[e].w;
        }

        cudaMemcpy(d_sourceVertices, sourceVertices, batchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_destinationVertices, destinationVertices, batchSize * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, weights, batchSize * sizeof(long long int), cudaMemcpyHostToDevice);

        startTime = rtClock();
        // Inserting batch of edges
        if(!(*negCycle))
        {
            thrust::device_ptr <int> src_ptr(d_sourceVertices);
            thrust::device_ptr<int> dst_ptr(d_destinationVertices);
            thrust::device_ptr<long long int> wgt_ptr(d_weights);

            thrust::sort_by_key(
                src_ptr,
                src_ptr + batchSize,
                thrust::make_zip_iterator(
                    thrust::make_tuple(dst_ptr, wgt_ptr)));
            
            cudaMemset(d_batchOutDegree, 0, (numNodes + 1) * sizeof(int));
            
            computeIndices <<< GRID_SIZE, BLOCK_SIZE >>> (d_batchOutDegree, d_sourceVertices, batchSize);
            cudaDeviceSynchronize();

            
            // Performing prefix sum using thrust library
            thrust::device_ptr<int> batchOutDegree_ptr(d_batchOutDegree);
            thrust::inclusive_scan(batchOutDegree_ptr, batchOutDegree_ptr + numNodes + 1, batchOutDegree_ptr);

            /*
            // Doing prefix sum using CUB library
            size_t required_temp_storage_bytes = 0;
            cub::DeviceScan::InclusiveSum(
                nullptr,
                required_temp_storage_bytes,
                d_batchOutDegree,
                d_batchOutDegree,
                numNodes + 1);
            if(required_temp_storage_bytes > tempStorageBytes)
            {
                printf("Warning: temp storage too small! Required = %zu bytes, Actual = %zu\n", required_temp_storage_bytes, tempStorageBytes);
                exit(EXIT_FAILURE);
            }
            cub::DeviceScan::InclusiveSum(
                d_tempStorage,
                required_temp_storage_bytes,
                d_batchOutDegree,
                d_batchOutDegree,
                numNodes + 1);
            */

            bool relaxed;

            cudaMemset(d_relaxed, false, sizeof(bool));

            // Insert batch of edges
            dim3 BLOCK_SIZE(32, 16);
            // Inside loop: One warp for each vertex, (one warp is benficial because in this case we are inserting a batch of edges and not adll edges at once, in a batch of size k there will be less number of edges adding to a single vertex than in the case where we are adding all edges at once. In static graph setting a block of 4 warps is dedicated to each vertex each time in loop.
            insertBatchOfEdgesKernel <<< 256, BLOCK_SIZE >>> (numNodes, d_destinationVertices, d_weights, d_batchOutDegree, d_adjListSizes, d_adjListCapacities, d_adjListDestinationVertices, d_adjListWeights, d_relaxed, d_flagA, d_dist);
            
            cudaMemcpy(&relaxed, d_relaxed, sizeof(bool), cudaMemcpyDeviceToHost);

            if(relaxed)
            {
                runPartialSSSP();
            }
        }
        endTime = rtClock();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }

        timeConsumed = (endTime - startTime) * 1e3;

        printf("-----BATCH INSERT-----\n");
        printf("batchSize,timeConsumed(ms)\n");
        printf("%d,%.6f\n", batchSize, timeConsumed);

        cudaFree(d_sourceVertices);
        cudaFree(d_destinationVertices);
        cudaFree(d_weights);
        free(sourceVertices);
        free(destinationVertices);
        free(weights);
    }

    void batchQuery(int batchSize, int* vertices)
    {
        printf("-----BATCH QUERY-----\n");
        if(*negCycle)
        {
            printf("Negative-cycle detected!\n");
            return;
        }

        cudaMemcpy(dist, d_dist, numNodes * sizeof(long long int), cudaMemcpyDeviceToHost);
        printf("vertex,shortest-distance\n");
        for(int k = 0; k < batchSize; k++)
        {
            int u = vertices[k];

            if(dist[u] == LLONG_MAX)
            {
                printf("%d,INF\n",u); 
            }
            else
            {
                printf("%d,%lld\n", u, dist[u]);
            }
        }
    }

    void batchQueryAllVertices()
    {
        printf("-----BATCH QUERY-----\n");
        if(*negCycle)
        {
            printf("Negative-cycle detected!\n");
            return;
        }

        cudaMemcpy(dist, d_dist, numNodes * sizeof(long long int), cudaMemcpyDeviceToHost);
        printf("vertex,shortest-distance\n");
        for(int u = 0; u < numNodes; u++)
        {
            if(dist[u] == LLONG_MAX)
            {   
                printf("%d,INF\n",u); 
            }
            else
            {
                printf("%d,%lld\n", u, dist[u]);
            }   
        } 
    }
};

void solve(int numNodes, int sourceVertex, int numQueries, Queries* inputQueries) 
{
    GraphSSSP* G = new GraphSSSP(numNodes, sourceVertex);
    for(int query = 0; query < numQueries; query++)
    {
        if(inputQueries[query].command == '+')
        {
            G->batchInsert(inputQueries[query].n, inputQueries[query].data.edgesToAdd);
        }
        else if(inputQueries[query].n == -1)
        {
            G->batchQueryAllVertices();
        }else
        {
            G->batchQuery(inputQueries[query].n, inputQueries[query].data.verticesToKnow);
        }
    }

    delete G;
}
