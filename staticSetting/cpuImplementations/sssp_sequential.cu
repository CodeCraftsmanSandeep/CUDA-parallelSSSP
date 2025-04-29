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

void SSSP(const int numNodes, const int numEdges, Edge* edgeList, int sourceVertex, long long int* dist, bool* negCycle);

void solve(int numNodes, int numEdges, Edge* edgeList, int sourceVertex) 
{ 
    // dist[u] -> shortest distance of u from sourceVertex
    long long int* dist = (long long int*)malloc(numNodes * sizeof(long long int));
    bool* negCycle = (bool*)malloc(sizeof(bool));

    std::vector <double> executionTimes(NUM_RUNS);

    double startTime, endTime;

    for(int run = 0; run < NUM_RUNS; run++){
        // Start time
        startTime = rtClock();

        // Compute the colution
        SSSP(numNodes, numEdges, edgeList, sourceVertex, dist, negCycle);

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

    free(negCycle);
    free(dist);
    return;
}

void SSSP(const int numNodes, const int numEdges, Edge* edgeList, int sourceVertex, long long int* dist, bool* negCycle)
{
    // Initilializing distance
    for(int u = 0; u < numNodes; u++)
    {
        dist[u] = LLONG_MAX;
    }

    // Setting sourceVertex shortest distance to 0
    dist[sourceVertex] = 0;

    bool relaxed = true;
    int u, v;
    long long int distCandidate;

    // Iterating atmost |V| times
    for(int iter = 1; (iter <= numNodes) && (relaxed); iter++)
    {
        relaxed = false;
        for(int e = 0; e < numEdges; e++)
        {
            // Relax edge if possible
            u = edgeList[e].u;
            v = edgeList[e].v;

            if(dist[u] != LLONG_MAX)
            {
                distCandidate = dist[u] + edgeList[e].w;
                if(distCandidate < dist[v])
                {
                    dist[v] = distCandidate;
                    relaxed = true;
                }
            }
        }
    }

    // Detect negative weight cycle
    *negCycle = relaxed;

    return;
}
