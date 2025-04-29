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


void SSSP(int numNodes, int numEdges, Edge* edgeList, int sourceVertex, long long int* dist, bool* negCycle, bool* flagA, bool* flagB);

void solve(int numNodes, int numEdges, Edge* edgeList, int sourceVertex) 
{ 
    // dist[u] -> shortest distance of u from sourceVertex
    long long int* dist = (long long int*)malloc(numNodes * sizeof(long long int));
    bool* negCycle = (bool*)malloc(sizeof(bool));


    bool* flagA = (bool*)malloc(numNodes * sizeof(bool));
    bool* flagB = (bool*)malloc(numNodes * sizeof(bool));

    std::vector <double> executionTimes(NUM_RUNS);

    double startTime, endTime;

    for(int run = 0; run < NUM_RUNS; run++){
        // Start time
        startTime = rtClock();

        SSSP(numNodes, numEdges, edgeList, sourceVertex, dist, negCycle, flagA, flagB);

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
    return;
}

void SSSP(const int numNodes, const int numEdges, Edge* edgeList, int sourceVertex, long long int* dist, bool* negCycle, bool* flagA, bool* flagB)
{
    // Setting negative cycle to false
    *negCycle = false;

    // Initilializing distance
    for(int u = 0; u < numNodes; u++){
        dist[u] = LLONG_MAX;
        flagA[u] = false;
        flagB[u] = false;
    }

    // Setting sourceVertex shortest distance to 0
    dist[sourceVertex] = 0;
    flagA[sourceVertex] = true;

    bool relaxed = false;
    int u, v, e;
    long long int w, distCandidate;

    // Iterating ((numNodes + 1) / 2 - 1) * 2 times
    for(int iter = 1; iter < ((numNodes + 1) >> 1); iter++)
    {
        for(e = 0; e < numEdges; e++)
        {
            // Relax edge if possible
            u = edgeList[e].u;
            if(flagA[u] == false) continue;
                
            v = edgeList[e].v;
            w = edgeList[e].w;
            distCandidate = dist[u] + w;
            if(distCandidate < dist[v])
            {   
                dist[v] = distCandidate;
                flagB[v] = true;
                relaxed = true;
            }
        }

        if(!relaxed) return;
        
        relaxed = false;
        for(u = 0; u < numNodes; u++) flagA[u] = false;

        for(e = 0; e < numEdges; e++)
        {
             u = edgeList[e].u;
             if(flagB[u] == false) continue;
 
             v = edgeList[e].v;
             w = edgeList[e].w;
             distCandidate = dist[u] + w;
             if(distCandidate < dist[v])
             {
                 dist[v] = distCandidate;
                 flagA[v] = true;
                 relaxed = true;
             }
        }
        if(!relaxed) return;
        
        relaxed = false;
        for(u = 0; u < numNodes; u++) flagB[u] = false;
    }

    if((numNodes & 1) == 0)
    {
        for(e = 0; e < numEdges; e++)
        {
            // Relax edge if possible
            u = edgeList[e].u;
            if(flagA[u] == false) continue;

            v = edgeList[e].v;
            w = edgeList[e].w;
            distCandidate = dist[u] + w;
            if(distCandidate < dist[v])
            {
                dist[v] = distCandidate;
                flagB[v] = true;
                relaxed = true;
            }
        }

        if(!relaxed) return;

        // Detect negative weight cycle
        for(e = 0; e < numEdges; e++)
        {
             u = edgeList[e].u;
             if(flagB[u] == false) continue;

             v = edgeList[e].v;
             w = edgeList[e].w;
             distCandidate = dist[u] + w;
             if(distCandidate < dist[v])
             {
                 *negCycle = true;
                 return;
             }
        }
    }else{
        // Detect negative weight cycle
        for(e = 0; e < numEdges; e++)
        {
             u = edgeList[e].u;
             if(flagA[u] == false) continue;

             v = edgeList[e].v;
             w = edgeList[e].w;
             distCandidate = dist[u] + w;
             if(distCandidate < dist[v])
             {   
                 *negCycle = true;
                 return;
             }
        }
    }

    return;
}
