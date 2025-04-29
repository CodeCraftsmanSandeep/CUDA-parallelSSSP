#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <vector>
#include <algorithm>
#include <numeric>

double rtClock(){
    struct timezone Tzp;
    struct timeval Tp;
    int stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

// Directed edge
struct Edge{
    int u, v;
    long long int w;
};

double findMean(const std::vector <double> times){
    if(times.size() == 0) return 0;

    return std::accumulate(times.begin(), times.end(), (double)0) / times.size();
}

double findMedian(std::vector <double> times){
    if(times.size() == 0) return 0;

    std::sort(times.begin(), times.end());
    if(times.size() & 1) return times[times.size()/2];
    return (times[times.size()/2 - 1] + times[times.size()/2])/2;
}

double findStandardDeviation(const std::vector<double> times) {
    if(times.size() <= 1) return 0;

    // Calculate mean
    double mean = findMean(times);

    // Calculate variance
    double variance = 0.0;
    for (double x : times) {
        variance += (x - mean) * (x - mean);
    }
    variance /= (times.size() - 1);

    // Return standard deviation
    return std::sqrt(variance);
}

void solve(int, int, Edge*, int);

int main(){
    // numNodes -> Number of nodes
    // numEdges -> Number of edges
    int numNodes, numEdges;
    scanf("%d%d", &numNodes, &numEdges);

    // Scanning m edges
    Edge* edgeList = (Edge*)malloc(numEdges * sizeof(Edge));
    for(int e = 0; e < numEdges; e++){
        int u, v;
        long long int w;
        scanf("%d%d%lld", &u, &v, &w);

        edgeList[e].u = u;
        edgeList[e].v = v;
        edgeList[e].w = w;
    }

    // sourceVertex -> Source vertex
    int sourceVertex;
    scanf("%d", &sourceVertex);

    solve(numNodes, numEdges, edgeList, sourceVertex);
    return 0;
}
