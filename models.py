import numpy as np

def logx(x, a, b, c):
    return a*np.log(x*b)+c
    
def expx(x, a, b, c):
    return c+(a/np.exp(b*x))

def model_1(x, a, b, c, d, e):
    n, k = x
    tb = b*n
    te = e*k
    td = d/np.exp(te)
    ta = a*np.log(tb)
    tc = c+ta+td
    return tc
    
def M_1(n, k):
    a = 0.41681823
    b = 0.85906102
    c = -0.47034162
    d = 4.39147762
    e = 0.31129515
    return model_1((n,k), a, b, c, d, e)


def S_1(n, k):
    a = 0.03452225
    b = 0.92507561
    c = 0.18967811
    d = 1.22202939
    e = 0.30130025
    return model_1((n,k), a, b, c, d, e)


def model_2(x, a, b, c, d, e, f, g):
    n, k = x
    bn = b*n
    ck = c*k
    en = e*n
    albn = a*np.log(bn)
    dlen = d*np.log(en)
    eck = np.exp(ck)
    t1 = albn/eck
    t2 = dlen
    t3 = f/eck
    t4 = g
    return t1+t2+t3+t4
    
    
def M_2(n, k):
    a = 0.5950798
    b = 2.13501329
    c = 0.31483198
    d = 0.34168386
    e = 1.62626029
    f = 0.24154007
    g = -0.22447648
    return model_2((n,k), a, b, c, d, e, f, g)


def S_2(n, k):
    a = -6.40196224e-02
    b = 1.00278958e+00
    c = 2.97982577e-01
    d = 4.32116618e-02
    e = 3.47735365e+01
    f = 1.61551054e+00
    g = -2.21250089e-02
    return model_2((n,k), a, b, c, d, e, f, g)
    
    
def model_3(x, a, b, c, d):
    n, k = x
    bn = b*n
    ck = c*k
    albn = a*np.log(bn)
    eck = np.exp(ck)
    t1 = albn/eck
    t2 = d
    return t1+t2


def M_3(n, k):
    a = 1.08329634
    b = 0.04169329
    c = 0.15497249
    d = 1.68079096
    return model_3((n,k), a, b, c, d)


def S_3(n, k):
    a = 5.69262827e-02
    b = 3.07801334e+06
    c = 2.95075272e-01
    d = 4.00871794e-01
    return model_3((n,k), a, b, c, d)
    
    
def model_4(x, xa, ya, xb, yb, xc, yc, ax, bx, ay, by, az, bz):
    n, k = x
    t1 = xa*np.log(xb*n)/np.exp(ya*k)
    t2 = xa*yb*k/np.exp(ya*k)
    t3 = xc/np.exp(yc*k)
    t4 = ax*np.log(bx*n)/np.power(by*n, ay*k)
    t5 = az*np.log(bz*n)
    return t1-t2+t3+t4+t5


def M_4(n, k):
    xa = 6.45905993e-01
    ya = 3.52109407e-01
    xb = 5.07293999e-09
    yb = -3.01350292e+00
    xc = 1.62610687e+01
    yc = 6.23341960e-01
    ax = 9.36642918e-01
    bx = 9.84404369e-01
    ay = 1.49133420e+00
    by = 1.14214126e+00
    az = 3.46738306e-01
    bz = 5.78073547e-01
    return model_4((n,k), xa, ya, xb, yb, xc, yc, ax, bx, ay, by, az, bz)


Ms = [M_1, M_2, M_3, M_4]
Ss = [S_1, S_2, S_3]

def prediction(n, k):
    return (M_2(n,k),S_1(n,k))