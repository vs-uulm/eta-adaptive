from statistics import mean
from igraph import Graph
from scipy.stats import norm
from helpers import frequencies
import numpy as np


# Number of runs per parameter combination
runs = range(100)

# network sizes to generate
noderange = list(range(50, 10001, 50))

# number of edges each node generates when being added to the network
edgerange = range(2, 25, 2)

alldata = []

try:
    for nodes in noderange:
        for edges in edgerange:
            if edges >= nodes:
                break
            ms = []
            ss = []
            for run in runs:
                freq = frequencies(nodes, edges)
                x = list(range(len(freq)))

                # use fixed size frequency representation
                expandeddata = np.repeat(x, np.array([int(_ * 10000) for _ in freq]).astype(int))

                m, s = norm.fit(expandeddata)
                ms.append(m)
                ss.append(s)
            alldata.append((nodes, param, ms, ss))
            # progress indication
            print(nodes, param, mean(ms), mean(ss))
except KeyboardInterrupt:
    # allows storing of results after interrupt through ctrl+c
    pass
    
from datetime import datetime
import pickle
pickle.dump(alldata, open("normal_approx_"+datetime.now().strftime("%Y-%m-%d-%H_%M_%S")+".p", "wb"))