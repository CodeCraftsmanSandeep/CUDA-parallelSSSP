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

// Queries of type (adding edge '+' (or) to know distances of vertices '?')
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

void solve(int, int, int, Queries*);

int main(){
    // numNodes -> Number of nodes
    int numNodes;
    scanf("%d", &numNodes);

    // sourceVertex -> Source vertex
    int sourceVertex;
    scanf("%d", &sourceVertex);

    // numQueries ~> Number of queries
    int numQueries;
    scanf("%d", &numQueries);

    Queries* inputQueries = new Queries[numQueries];
    for(int i = 0; i < numQueries; i++)
    {
        char ch;
        // Read next non-whitespace character
        do {
            ch = getchar();
        } while (ch == ' ' || ch == '\n');

        inputQueries[i].command = ch;
        scanf("%d", &(inputQueries[i].n));
        if(inputQueries[i].command == '+')
        {
            inputQueries[i].data.edgesToAdd = (Edge*)malloc(inputQueries[i].n * sizeof(Edge));
            for(int j = 0; j < inputQueries[i].n; j++)
            {
                scanf("%d %d %lld", &(inputQueries[i].data.edgesToAdd[j].u),&(inputQueries[i].data.edgesToAdd[j].v), &(inputQueries[i].data.edgesToAdd[j].w));
            }
        }
        else if(inputQueries[i].command == '?')
        {
            if(inputQueries[i].n == -1)
            {
                inputQueries[i].data.verticesToKnow = nullptr;
            }
            else
            {
                inputQueries[i].data.verticesToKnow = (int*)malloc(inputQueries[i].n * sizeof(int));
                for(int j = 0; j < inputQueries[i].n; j++)
                {
                    scanf("%d", &(inputQueries[i].data.verticesToKnow[j]));
                }
            }
        }
        else
        {
            printf("Invalid command: %c\n", inputQueries[i].command);
            return 1; // Failure
        }
    }

    solve(numNodes, sourceVertex, numQueries, inputQueries);
    return 0;
}
