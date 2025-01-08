import pandas as pd
import numpy as np
import qutip as qt
from datetime import datetime
import asyncio

def background(f):
    def wrapped(*args, **kwargs):
        return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    return wrapped

def g2(pops):
    num = np.sum(np.array([k*(k-1) for k in range(len(pops))])*pops)
    den = (np.sum(np.arange(len(pops))*pops))**2
    return num/den

# all units MHz
fr = 5230.2
fq, g, lamb = 5313.25, 17, 3.640
# fq, g, lamb = 5313.25,  0, 0.000 # no witness
# fq, g, lamb = 5348.50, 17, 2.500 # early set of measurements
Nc = 7
Nq = 2
Nb = 2
a = qt.destroy(Nc)
b = qt.destroy(Nq)
c = qt.destroy(Nb)
kappa = 0.1
gamma = 0.1
gammab = 0.1
aq = 227
ab = 263
gb = 13.7

def make_H(lamb, dfreq, damp, Ne, detune=0):
    H = 0
    # cavity energy
    op_list = [a.dag()*a, qt.qeye(Nq)]
    for emit in range(Ne):
        op_list.append(qt.qeye(Nb))
    H += (fr + lamb - dfreq)*qt.tensor(op_list)
    # witness qubit energy
    op_list = [qt.qeye(Nc), b.dag()*b]
    for emit in range(Ne):
        op_list.append(qt.qeye(Nb))
    H += (fq - lamb - dfreq)*qt.tensor(op_list)
    # witness qubit-cavity coupling
    op_list = [a, b.dag()]
    for emit in range(Ne):
        op_list.append(qt.qeye(Nb))
    H += g*qt.tensor(op_list)
    op_list = [a.dag(), b]
    for emit in range(Ne):
        op_list.append(qt.qeye(Nb))
    H += g*qt.tensor(op_list)
    # witness qubit anharmonicity
    op_list = [qt.qeye(Nc), b.dag()*b.dag()*b*b]
    for emit in range(Ne):
        op_list.append(qt.qeye(Nb))
    H += (-aq/2)*qt.tensor(op_list)
    # cavity drive
    op_list = [a.dag() + a, qt.qeye(Nq)]
    for emit in range(Ne):
        op_list.append(qt.qeye(Nb))
    H += damp*qt.tensor(op_list)
    # emitter energy
    if Ne > 0:
        for emit1 in range(Ne):
            op_list = [qt.qeye(Nc), qt.qeye(Nq)]
            for emit2 in range(Ne):
                if emit1 == emit2:
                    op_list.append(c.dag()*c)
                else:
                    op_list.append(qt.qeye(Nb))
            H += (detune + fr - dfreq)*qt.tensor(op_list)
    # emitter-cavity coupling
    if Ne > 0:
        for emit1 in range(Ne):
            op_list = [a, qt.qeye(Nq)]
            for emit2 in range(Ne):
                if emit1 == emit2:
                    op_list.append(c.dag())
                else:
                    op_list.append(qt.qeye(Nb))
            H += gb*qt.tensor(op_list)
    if Ne > 0:
        for emit1 in range(Ne):
            op_list = [a.dag(), qt.qeye(Nq)]
            for emit2 in range(Ne):
                if emit1 == emit2:
                    op_list.append(c)
                else:
                    op_list.append(qt.qeye(Nb))
            H += gb*qt.tensor(op_list)
    # emitter anharmonicity
    if Ne > 0:
        for emit1 in range(Ne):
            op_list = [qt.qeye(Nc), qt.qeye(Nq)]
            for emit2 in range(Ne):
                if emit1 == emit2:
                    op_list.append(c.dag()*c.dag()*c*c)
                else:
                    op_list.append(qt.qeye(Nb))
            H += (-ab/2)*qt.tensor(op_list)

    return H

def make_cops(Ne):
    c_ops = []
    # cavity decay
    op_list = [a, qt.qeye(Nq)]
    if Ne > 0:
        for emit in range(Ne):
            op_list.append(qt.qeye(Nb))
    c_ops.append(np.sqrt(kappa)*qt.tensor(op_list))
    # witness qubit decay
    op_list = [qt.qeye(Nc), b]
    if Ne > 0:
        for emit in range(Ne):
            op_list.append(qt.qeye(Nb))
    c_ops.append(np.sqrt(gamma)*qt.tensor(op_list))
    # blockade qubit decay
    if Ne > 0:
        for emit1 in range(Ne):
            op_list = [qt.qeye(Nc), qt.qeye(Nq)]
            for emit2 in range(Ne):
                if emit1 == emit2:
                    op_list.append(c)
                else:
                    op_list.append(qt.qeye(Nb))
            c_ops.append(np.sqrt(gammab)*qt.tensor(op_list))
    
    return c_ops

# compute steady state over range of powers
Ne = 1
powers = kappa*np.logspace(np.log10(0.44574034758115627), np.log10(70.64508424866254), 45)
cops = make_cops(Ne)

H0 = make_H(lamb, 0, 0, Ne)
fd = H0.eigenstates()[0][1] - H0.eigenstates()[0][0]
print(fd)

@background
def find_ss(power):
    H = make_H(lamb, fd, power, Ne, detune=0)
    rho = qt.steadystate(H, cops)
    rho = rho.ptrace(0)
    x = np.abs(np.diag(rho.full()))
    return x

bsize = 50
nbatches = int(len(powers)/bsize) + 1
for n in range(nbatches):
    loop = asyncio.get_event_loop()                                              
    looper = asyncio.gather(*[find_ss(p) for p in powers[n*bsize:(n+1)*bsize]])
    if n==0:
        data = loop.run_until_complete(looper)
    else:
        data += loop.run_until_complete(looper)                          

data = np.array(data)

ncol = []
pcol = []
dcol = []
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        ncol.append(j)
        pcol.append(powers[i])
        dcol.append(data[i,j])

df_dict = {
        'n': ncol,
        'power': pcol,
        'data': dcol
    }
df = pd.DataFrame(df_dict)
df.to_csv('N0-ss-test5.csv')