import pandas as pd
import matplotlib.pyplot as plt
from scipy.fftpack import fft
import numpy as np

def Evaluate(filename):
    dbase = pd.read_pickle("input.pickle")
    d     = pd.read_csv("output/%s.dat"%filename)

    Ninput     = 28*28
    KinputFrom = 0
    KinputTo   = KinputFrom + Ninput

    Noutput     = 10
    KoutputFrom = KinputTo
    KoutputTo   = KoutputFrom + Noutput

    Nhidden     = 10+10+10
    KhiddenFrom = KoutputTo
    KhiddenTo   = KhiddenFrom + Nhidden

    dAns = dbase.ix[:,1:2]
    #print dAns
    #dAns.columns = ["iSample","Answer"]
#Answer.columns = ["Answer"]
    AnsYY   = dbase.ix[:,KoutputFrom+2:KoutputTo+2] # +2 <- index and answer
    AnsYY.columns=["A%d"%s for s in range(0,10)]
    AnsY    = pd.merge(dAns, AnsYY, left_index=True, right_index=True)
    #print AnsY

#EvaY = (d[(d.totalIter%5000)>4000].groupby("sampleIndex").mean())[["X(%d)"%s for s in range(KoutputFrom,KoutputTo)]]
    EvaY = (d[d.comment=="Evaluate"].groupby("sampleIndex").mean())[["X(%d)"%s for s in range(KoutputFrom,KoutputTo)]]
    EvaY.columns = ["E%d"%s for s in range(0,10)]
    EvaY[EvaY>=0]=+1
    EvaY[EvaY<0] =-1

    #print EvaY

    exam = pd.concat([AnsY,EvaY],axis=1,join="inner")
#exam.to_pickle("exam2.pickle")
    #print exam

    def Check(x):
        Ecount = 0
        Egood  = False
        for d in range(10):
            if x["E%d"%d]==+1: Ecount += 1
            if x["A%d"%d]==+1 and x["E%d"%d]==+1: Egood = True
        if(Egood):
            if   Ecount == 1: return 0
            else            : return 1
        else:
            if   Ecount == 0: return 2
            else            : return 3

    exam["flag"] = exam.apply(Check,axis=1)
    #exam.to_pickle("exam3.pickle")
    #print exam
    #print exam.groupby("flag").count()["E0"]
    flag0 = exam[exam.flag==0].count()["flag"]
    flag1 = exam[exam.flag==1].count()["flag"]
    flag2 = exam[exam.flag==2].count()["flag"]
    flag3 = exam[exam.flag==3].count()["flag"]

    print filename,(flag0, flag1, flag2, flag3)

Evaluate("out_100_10_0.01000")







"""
Evaluate("out_10000_5_0.00100")
Evaluate("out_10000_5_0.01000")
Evaluate("out_10000_5_0.10000")

Evaluate("out_1000_10_0.00100")
Evaluate("out_1000_10_0.01000")
Evaluate("out_1000_10_0.10000")

Evaluate("out_1000_50_0.00100")
Evaluate("out_1000_50_0.01000")
Evaluate("out_1000_50_0.10000")

Evaluate("out_100_100_0.00100")
Evaluate("out_100_100_0.01000")
Evaluate("out_100_100_0.10000")
"""
