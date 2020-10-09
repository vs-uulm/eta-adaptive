from statistics import mean
import pickle
import numpy as np
from igraph import Graph


def discretize_pointwise(dist, start=0, threshold=0.01, minvalue=0.95):
    a = []
    i = start
    while True:
        a.append(dist.pdf(i))
        i += 1
        if dist.pdf(i) < threshold and sum(a) > minvalue:
            return a/sum(a)

def discretize_intervals(dist, start=0, threshold=0.01, halfstart=True):
    a = []
    i = start+(0.5 if halfstart else 0)
    prev = 0
    while True:
        cur = dist.cdf(i)
        a.append(cur-prev)
        prev = cur
        if 1-cur < threshold:
            a.append(1-dist.cdf(i))
            break
        i+=1
    return a


def frequencies(nodes, edges):
    g = Graph.Establishment(nodes, edges, [1], [[1]])
    spl = g.shortest_paths_dijkstra()
    return [sum([ll.count(_) for ll in spl])/(nodes*nodes) for _ in range(max(map(max, spl)))]


def gen_testcase(nodes, edges):
    testcase = {}
    freq =  frequencies(nodes,edges)
    testcase["freq"] = freq
    # baseic x range
    testcase["x"] = list(range(len(freq)))
    # expanded x range to see beyond the edge
    testcase["x2"] = list(range(len(freq)+5))
    # numpy x range
    testcase["x3"] = np.linspace(0, testcase["x2"][-1])
    # expanded numpy x range
    testcase["x4"] = np.linspace(0, len(freq)+5)
    testcase["expandeddata"] = np.repeat(testcase["x"][1:], np.array([int(_*10000) for _ in freq[1:]]).astype(int))
    return testcase
    

def load_data(picklename, lowfilter=1):
    base_data =  [(_[0], _[1], _[2], _[3]) for _ in pickle.load(open(picklename, "rb")) if _[0] >= lowfilter]
    data = {
        "nodes": [_[0] for _ in base_data],
        "edges": [_[1] for _ in base_data],
        "mu_min": [min(_[2]) for _ in base_data],
        "mu_means": [mean(_[2]) for _ in base_data],
        "mu_max": [max(_[2]) for _ in base_data],
        "si_min": [min(_[3]) for _ in base_data],
        "si_means": [mean(_[3]) for _ in base_data],
        "si_max": [max(_[3]) for _ in base_data],
        "unique_nodes": list(set([_[0] for _ in base_data]))
    }
    return data

    
def ecdf(x):
    c = 1.0/len(x)
    return np.arange(c, 1.0 + c, c)
    
    
def e_survival(x):
    y = 1.0 - ecdf(x)
    y[-1] = np.nan
    return y
    
    
def asqe(data, model, dimension="mu_means"):
    error = []
    for n,e,v in zip(data["nodes"], data["edges"], data[dimension]):
        error.append((model(n, e)-v)**2)
    return error

