// Program to find Dijkstra's shortest path using 
// priority_queue in STL
// Complie this file using this command
// c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

#include<bits/stdc++.h>
using namespace std;
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

// iPair ==> Integer Pair 
typedef pair<int , double> iPair;

// To add an edge 
void addEdge(vector <pair<int, double > > adj[], int u, int v, double wt)
{
    adj[u].push_back(make_pair(v, wt));
    adj[v].push_back(make_pair(u, wt));
}


// Prints shortest paths from src to all other vertices 
vector<double> shortestPath(vector<pair<int,double > > adj[], int num_vert, int src)
{

    priority_queue< iPair, vector <iPair> , greater<iPair> > pq;
    vector<double> dist(num_vert, numeric_limits<double>::infinity());
    pq.push(make_pair(0, src));
    dist[src] = 0;

    while (!pq.empty())
    {
        int u = pq.top().second;
        pq.pop();

        for (auto x : adj[u])
        {
            int v = x.first;
            double weight = x.second;
            if (dist[v] > dist[u] + weight)
            {
                dist[v] = dist[u] + weight;
                pq.push(make_pair(dist[v], v));
            }
        }
    }
    return dist;
//    printf("Vertex Distance from Source\n");
//    for (int i = 0; i < V; ++i)
//        printf("%d \t\t %d\n", i, dist[i]);
}


py::array_t<double> get_distance_m(int num_vert, py::array_t<double> src, py::array_t<double> trg, py::array_t<double> wt)
{
    py::array_t<double> results = py::array_t<double>(num_vert * num_vert);
    py::buffer_info results_buf = results.request();
    double *results_ptr = (double *) results_buf.ptr;
    vector<iPair > adj[num_vert];
    py::buffer_info src_buf = src.request();
    py::buffer_info trg_buf = trg.request();
    py::buffer_info wt_buf = wt.request();
    double *src_ptr = (double *) src_buf.ptr;
    double *trg_ptr = (double *) trg_buf.ptr;
    double *wt_ptr = (double *) wt_buf.ptr;

    size_t num_edegs = src_buf.shape[0];

    for (size_t idx = 0; idx < num_edegs; idx++){

        addEdge(adj, int (src_ptr[idx]), int(trg_ptr[idx]), wt_ptr[idx]);
    }
    for (int i = 0; i < num_vert; i++) {

        vector<double> dist = shortestPath(adj, num_vert, i);
        for (int j = 0; j < num_vert; j++) {
            results_ptr[i * num_vert + j] = dist[j];
        }

    }
    results.resize({num_vert, num_vert});
    return results;
}

PYBIND11_MODULE(distance_mat, m) {
m.doc() = "pybind11 example plugin"; // optional module docstring

m.def("get_distance_m", &get_distance_m, "A function");
}
