from statistics import mean
import pickle
import numpy as np
from igraph import Graph
from scipy.special import binom 
import math


def packaged_coupons_expected(n, c, a):
    n_ov_c = binom(n,c)
    logc_na = math.ceil(math.log(n-a,c))
    #print(f"log_{c}({n}-{a}) = {logc_na}")
    #print(f"({n} {c}) = n_ov_c")
    tmp = 0
    for j in range(logc_na):
        a_ov_j = binom(a,j)
        t1 = (-1)**(logc_na-j+1)
        t2 = binom(n,c) - binom(n-a+j,c)
        t3 = binom(a-j-1, a-logc_na)
        tmp += t1/t2*a_ov_j*t3
    return n_ov_c*tmp

def tree_depth_bound(n, eta, beta):
    z = packaged_coupons_expected(n,eta,n*beta)
    return math.log(1-(z-1)*(1-eta)/(eta+1), eta)+1


def determine_pt(n, k):
    from models import prediction
    from scipy. stats import norm
    mu, si = prediction(n,k)
    dist = norm(mu, si)
    
    prepared_values = [1/n, k*(2*n-k-1)/n**2]
    f = discretize_pointwise(dist, startvalues=prepared_values)
    
    from functools import lru_cache
    
    @lru_cache(maxsize=None)
    def ffrak(i,t):
        if i >= t:
            return 0.0 # virtual extension of the probabilities
        return f[i]/sum(f[:t]) # sum(f[:t]) = sum of f(0), f(1), ..., f(t-1)
        
    from IPython.display import display, Markdown
    
    @lru_cache(maxsize=None)
    def ffrak_prime(i,t):
        if t >= len(f): # len(f)-1 is the last desired state
            return ffrak(i,t)
        elif i>=t:
            return 0.0
        else:
            chi = sum([ffrak(j, t) - ffrak_prime(j, t) for j in range(i+1,t+1)])
            delta = ffrak_prime(t,t+1) - ffrak(i,t) + sum([ffrak_prime(j,t+1) - ffrak_prime(j, t) for j in range(i+1,t)])
            display(Markdown(f"$ f'_{t}({i}) = f_{t}(i) + max(\\left( \\sum_{{j={i+1} }}^{t}f_{t}(j)-f'_{t}(j)\\right),\\left(f'_{t+1}({t}) + \\sum_{{j={i+1} }}^{t-1} (f'_{t+1}(j) - f'_{t}(j)) - f_{t}({i})\\right)) = {ffrak(i,t)} + max({chi},{delta}) = = {ffrak(i,t)+max(chi,delta):.3f} $"))
            return ffrak(i,t) + max(chi,delta)
    
    @lru_cache(maxsize=None)
    def p(h,t,f):
        if t == 1:
            display(Markdown(f"$p_{t}({h}) = 1-f_{t+1}({h}) = {1-f(h,t+1)}$"))
            return 1-f(0,t+1)
        if h == 0:
            display(Markdown(f"$p_{t}({h}) = 1-\\frac{{f_{t+1}(0)}}{{f_{t}(0)}} = 1-\\frac{{{f(0,t+1)} }}{{ {f(0,t)}}} = {1-(f(0,t+1)/f(0,t))}$\n"))
            return 1-(f(0,t+1)/f(0,t))
        else:
            display(Markdown(f"$p_{t}({h}) = 1- \\frac{{f_{t+1}({h}) - p_{t}({h-1}) * f_{t}({h-1}) }} {{ f_{t}({h}) }} = 1-\\frac{{ ({f(h,t+1)} - {p(h-1,t,f)} * {f(h-1,t)}) }}{{ {f(h,t)} }} = {1-(f(h,t+1) - p(h-1,t,f) * f(h-1,t))/f(h,t)}$\n"))
            return 1-(f(h,t+1) - p(h-1,t,f) * f(h-1,t))/f(h,t)
    
    forig = []
    for t in range(len(f)+1):
        forig.append([ffrak(h,t) for h in range(t)])

    fprime = []
    for t in range(len(f)+1):
        fprime.append([ffrak_prime(h,t) for h in range(t)])
        
    pts = []
    for t in range(1, len(f)): # 1, 2, ..., len(f)-1
        pts.append(([p(h,t,ffrak_prime) for h in range(t)])) # p1(0), p2(0)+p2(1), ...
    
    return pts,forig,fprime


def discretize_pointwise(dist, startvalues=[], threshold=0.01, minvalue=0.95):
    a = []
    a.extend(startvalues)
    i = len(startvalues)
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
    
    
def bias(data, model, dimension="mu_means"):
    error = []
    for n,e,v in zip(data["nodes"], data["edges"], data[dimension]):
        error.append((model(n, e)-v))
    return error
    
    
def asqe(data, model, dimension="mu_means"):
    error = []
    for n,e,v in zip(data["nodes"], data["edges"], data[dimension]):
        error.append((model(n, e)-v)**2)
    return error

