import numpy as np
import distance_mat

num_vert = 9
src = np.array([0, 0, 1, 1, 2, 2, 2, 3, 3, 4, 5, 6, 6, 7])
trg = np.array([1, 7, 2, 7, 3, 8, 5, 4, 5, 5, 6, 7, 8, 8])
wt = np.array([4, 8, 8, 11, 7, 2, 4, 9, 14, 10, 2, 1, 6, 7])

res = distance_mat.get_distance_m(num_vert, src, trg, wt)

print(res)
#mopile the cpp file first
#c++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` example.cpp -o example`python3-config --extension-suffix`

