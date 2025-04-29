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

struct EdgeNode 
{
    int v;
    long long int w;
};

double findMean(const std::vector <double> times);

double findMedian(std::vector <double> times);

double findStandardDeviation(const std::vector<double> times);

constexpr unsigned int NUM_RUNS = 11;

void SSSP(int numNodes, int numEdges, EdgeNode** adjList, int sourceVertex, long long int* dist, bool* negCycle, int* queueA, bool* flagA, int* queueB, bool* flagB);

void solve(int numNodes, int numEdges, Edge* edgeList, int sourceVertex) 
{
    // Calculating out degree of every vertex
    int* outDegree = (int*)calloc(numNodes, sizeof(int));
    for(int e = 0; e < numEdges; e++){
        outDegree[edgeList[e].u]++;
    }

    // Creating adj list
    EdgeNode** adjList = (EdgeNode**)malloc(numNodes * sizeof(EdgeNode*));

    for(int u = 0; u < numNodes; u++){
        adjList[u] = (EdgeNode*)malloc((outDegree[u] + 1) * sizeof(EdgeNode));
        adjList[u][0].v = outDegree[u]; // storing size in first adjcent node
        outDegree[u] = 1;
    }

    // Add edges into adj list
    for(int e = 0; e < numEdges; e++)
    {
        int u = edgeList[e].u;
        int v = edgeList[e].v;
        long long int w = edgeList[e].w;

        adjList[u][outDegree[u]].v = v;
        adjList[u][outDegree[u]].w = w;

        outDegree[u]++;
    }

    // dist[u] -> shortest distance of u from sourceVertex
    long long int* dist = (long long int*)malloc(numNodes * sizeof(long long int));
    bool* negCycle = (bool*)malloc(sizeof(bool));

    bool* flagA = (bool*)malloc(numNodes * sizeof(bool));
    bool* flagB = (bool*)malloc(numNodes * sizeof(bool));

    int* queueA = (int*)malloc(numNodes * sizeof(int));
    int* queueB = (int*)malloc(numNodes * sizeof(int));

    std::vector <double> executionTimes(NUM_RUNS);

    double startTime, endTime;

    for(int run = 0; run < NUM_RUNS; run++){
        // Start time
        startTime = rtClock();

        SSSP(numNodes, numEdges, adjList, sourceVertex, dist, negCycle, queueA, flagA, queueB, flagB);

        // End time
        endTime = rtClock();

        executionTimes[run] = (endTime - startTime) * 1e3; // in milli-seconds
    }

    printf("Num-runs,Mean-time(ms),Median-time(ms),Std-deviation(ms)\n");
    printf("%d,%.6f,%.6f,%.6f\n", NUM_RUNS, findMean(executionTimes), findMedian(executionTimes), findStandardDeviation(executionTimes));

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

    free(dist);
    free(negCycle);
    free(flagA);
    free(flagB);
    free(queueA);
    free(queueB);

    return;
}

void SSSP(int numNodes, int numEdges, EdgeNode** adjList, int sourceVertex, long long int* dist, bool* negCycle, int* queueA, bool* flagA, int* queueB, bool* flagB)
{
    // Setting negative cycle to false
    *negCycle = false;

    // Initilializing distance
    for(int u = 0; u < numNodes; u++){
        dist[u] = LLONG_MAX;
        flagA[u] = false;
        flagB[u] = false;
    }

    // Initializing queue sizes
    int queueASize = 0;
    int queueBSize = 0;

    // Setting sourceVertex shortest distance to 0
    dist[sourceVertex] = 0;
    queueA[queueASize++] = sourceVertex;
    flagA[sourceVertex] = true;

    int u, v;
    long long int w, distCandidate;
    // Iterating ((numNodes + 1) / 2 - 1) * 2 times
    for(int iter = 1; (iter < ((numNodes + 1) >> 1)) && (queueASize > 0); iter++)
    {
        while(queueASize)
        {
            u = queueA[--queueASize];
            flagA[u] = false;

            int outDegree = adjList[u][0].v;

            for(int i = 1; i <= outDegree; i++)
            {
                // Relax edge if possible
                v = adjList[u][i].v;
                w = adjList[u][i].w;
                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {
                    dist[v] = distCandidate;
                    if(flagB[v] == false){
                        queueB[queueBSize++] = v;
                        flagB[v] = true;
                    }
                }
            }
        }

        while(queueBSize){
            u = queueB[--queueBSize];
            flagB[u] = false;
            int outDegree = adjList[u][0].v;

            for(int i = 1; i <= outDegree; i++)
            {
                // Relax edge if possible
                v = adjList[u][i].v;
                w = adjList[u][i].w;
                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {
                    dist[v] = distCandidate;
                    if(flagA[v] == false){
                        queueA[queueASize++] = v;
                        flagA[v] = true;
                    }
                }
            }
        }
    }

    if((numNodes & 1) == 0)
    {
        while(queueASize)
        {
            u = queueA[--queueASize];
            flagA[u] = false;
            int outDegree = adjList[u][0].v;

            for(int i = 1; i <= outDegree; i++)
            {
                // Relax edge if possible
                v = adjList[u][i].v;
                w = adjList[u][i].w;
                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {
                    dist[v] = distCandidate;
                    if(flagB[v] == false){
                        queueB[queueBSize++] = v;
                        flagB[v] = true;
                    }
                }
            }
        }

        // Detect negative weight cycle
        while(queueBSize)
        {
            u = queueB[--queueBSize];
            int outDegree = adjList[u][0].v;

            for(int i = 1; i <= outDegree; i++)
            {
                // Relax edge if possible
                v = adjList[u][i].v;
                w = adjList[u][i].w;
                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {
                    *negCycle = true;
                    return;
                }
            }
        }
    }else{
        // Detect negative weight cycle
        while(queueASize)
        {   
            u = queueA[--queueASize];
            int outDegree = adjList[u][0].v;

            for(int i = 1; i <= outDegree; i++)
            {
                // Relax edge if possible
                v = adjList[u][i].v;
                w = adjList[u][i].w;
                distCandidate = dist[u] + w;
                if(distCandidate < dist[v])
                {  
                    *negCycle = true;
                    return;
                }   
            }   
        }   
    }

    return;
}
