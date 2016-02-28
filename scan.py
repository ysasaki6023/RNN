import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

import sys
#d = pd.read_csv("output/out_100_10_0.01000.dat")
d = pd.read_csv(sys.argv[1])

Ninput     = 28*28
KinputFrom = 0
KinputTo   = KinputFrom + Ninput

Noutput     = 10
KoutputFrom = KinputTo
KoutputTo   = KoutputFrom + Noutput

Nhidden     = 10+00+10
#Nhidden     = 0
KhiddenFrom = KoutputTo
KhiddenTo   = KhiddenFrom + Nhidden

#print d

d = d[d.comment=="Evaluate2"]
print d[d.iStep==100]

Index  = d["totalIter"]
#Xinput  = d[["X(%d)"%i for i in range(KinputFrom , KinputTo )]]
Xoutput = d[["X(%d)"%i for i in range(KoutputFrom, KoutputTo)]]
Xhidden_title = ["X(%d)"%i for i in range(KhiddenFrom, KhiddenTo)]
Xhidden = d[["X(%d)"%i for i in range(KhiddenFrom, KhiddenTo)]]
Routput = d[["R(%d)"%i for i in range(KoutputFrom, KoutputTo)]]
Rhidden = d[["R(%d)"%i for i in range(KhiddenFrom, KhiddenTo)]]
#RhiddenSum = Rhidden.sum(axis=1)
#yf = fft(RhiddenSum)

#plt.plot(Index, pd.rolling_mean(Xinput,20))
plt.plot(Index, pd.rolling_mean(Xoutput,20))
#plt.plot(Index, pd.rolling_mean(Xhidden.ix[:, 10:20],100))
#plt.legend(Xhidden_title,prop={"size":5})
#plt.plot(Index, RhiddenSum)
#plt.plot(Index, np.abs(yf))
plt.show()
