import Classes as cl
from Classes import graphUtil as gUtil
from Classes import sumSpectra as sS
from Classes import nuclideSpectra as nS
from Classes import *
from Classes import redRelSpectra as RS
import re
import matplotlib.pyplot as plt
import math
import numpy as np
import csv
import glob
import sys
from array import array
import matplotlib.animation as animation
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)

plt.style.use('custom')

# hmEFNEOS=HuberMueller.getFine(sS.fisNEOS, fileName="HBEFBinned.csv", bins=list(np.arange(1.8, 8.0001, .0001)))
# hmEFRENO=HuberMueller.getFine(sS.fisRENO, fileName="HBEFBinned.csv", bins=list(np.arange(1.8, 8.0001, .0001)))
i=1
analyzedSpectra=Util.readDict(RS, filePath=r"READ FROM CODE\\NEOS+RENO analyzedData.csv")
NEOS, RENO=analyzedSpectra["NEOS"], analyzedSpectra["RENO"]
NEOS.rebin(HuberMueller.bins[i])
NEOS.renorm()
RENO.rebin(HuberMueller.bins[i])
RENO.renorm()

plt.plot(NEOS.E, NEOS.S, marker="^")
plt.xlabel("Energy [MeV]")
plt.ylabel("Spectra [MeV$^{-1}$]")
plt.show()

# hmNEOS=HuberMueller.convToSum(sS.fisNEOS)[i]
# hmRENO=HuberMueller.convToSum(sS.fisRENO)[i]
# sumNEOS=sS.reactSumSpect({"u5":.655, "u8":.072, "pu9":.235, "pu1":.038})
# sumRENO=sS.reactSumSpect(sS.fisRENO)
# sumNEOS.rebin(HuberMueller.bins[i])
# sumRENO.rebin(HuberMueller.bins[i])
# sumRENO.renorm()
# hmRENO.renorm()
# sumNEOS.renorm()
# hmNEOS.renorm()

# dict=Util.readDict(nuclideSpectra, filePath="READ FROM CODE\\NEOSVTEST.csv")
# sumNEOST=sS.sumNucl(dict, bins=np.arange(0, 10, .01))
# sumNEOST.rebin(HuberMueller.bins[i])
# sumNEOST.renorm()
# plt.plot(sumNEOS.E, sumNEOS.S)
# plt.plot(sumNEOST.E, sumNEOST.S)
# plt.show()
# """RENO STUFF"""
# # E=[2.10E+00,
# # 2.40E+00,
# # 2.70E+00,
# # 3.00E+00,
# # 3.30E+00,
# # 3.60E+00,
# # 3.90E+00,
# # 4.20E+00,
# # 4.50E+00,
# # 4.80E+00,
# # 5.10E+00,
# # 5.40E+00,
# # 5.70E+00,
# # 6.00E+00,
# # 6.30E+00,
# # 6.60E+00,
# # 6.90E+00,
# # 7.20E+00,
# # 7.50E+00,
# # 7.80E+00,
# # 8.10E+00,
# # 8.40E+00]

# # renoR=[4.16E-01,
# # 6.78E-01,
# # 7.76E-01,
# # 8.57E-01,
# # 9.16E-01,
# # 9.86E-01,
# # 1.02E+00,
# # 1.09E+00,
# # 1.15E+00,
# # 1.11E+00,
# # 1.11E+00,
# # 1.16E+00,
# # 1.17E+00,
# # 1.28E+00,
# # 1.30E+00,
# # 1.32E+00,
# # 1.42E+00,
# # 1.59E+00,
# # 1.59E+00,
# # 1.78E+00,
# # 2.44E+00,
# # 2.37E+00]

# # renoErr=[1.19E-02,
# # 1.20E-02,
# # 9.33E-03,
# # 8.85E-03,
# # 9.08E-03,
# # 9.24E-03,
# # 9.06E-03,
# # 9.94E-03,
# # 1.01E-02,
# # 9.15E-03,
# # 9.77E-03,
# # 1.23E-02,
# # 1.41E-02,
# # 1.52E-02,
# # 1.40E-02,
# # 1.66E-02,
# # 2.66E-02,
# # 3.60E-02,
# # 4.58E-02,
# # 1.16E-01,
# # 3.37E-01,
# # 4.17E-01]


# # Sum= [1.46E-01,
# # 6.14E-01,
# # 7.52E-01,
# # 8.43E-01,
# # 8.78E-01,
# # 9.76E-01,
# # 1.01E+00,
# # 1.07E+00,
# # 1.13E+00,
# # 1.14E+00,
# # 1.15E+00,
# # 1.19E+00,
# # 1.20E+00,
# # 1.29E+00,
# # 1.33E+00,
# # 1.33E+00,
# # 1.37E+00,
# # 1.53E+00,
# # 1.65E+00,
# # 1.57E+00,
# # 3.08E+00,
# # 1.64E+00,
# # 1.56E+00]

# # HM=[1.49E-01,
# # 6.10E-01,
# # 7.46E-01,
# # 8.29E-01,
# # 9.04E-01,
# # 9.73E-01,
# # 1.03E+00,
# # 1.06E+00,
# # 1.11E+00,
# # 1.15E+00,
# # 1.17E+00,
# # 1.19E+00,
# # 1.21E+00,
# # 1.24E+00,
# # 1.25E+00,
# # 1.31E+00,
# # 1.35E+00,
# # 1.44E+00,
# # 1.58E+00,
# # 1.73E+00,
# # 2.06E+00,
# # 2.26E+00,
# # 2.60E+00]

# # rem96Y=[1.46E-01,
# # 6.16E-01,
# # 7.55E-01,
# # 8.47E-01,
# # 8.82E-01,
# # 9.82E-01,
# # 1.02E+00,
# # 1.08E+00,
# # 1.15E+00,
# # 1.15E+00,
# # 1.17E+00,
# # 1.21E+00,
# # 1.22E+00,
# # 1.32E+00,
# # 1.35E+00,
# # 1.32E+00,
# # 1.32E+00,
# # 1.36E+00,
# # 1.65E+00,
# # 1.57E+00,
# # 3.08E+00,
# # 1.64E+00,
# # 1.56E+00]

# # rem92Rb=[1.46E-01,
# # 6.15E-01,
# # 7.54E-01,
# # 8.45E-01,
# # 8.81E-01,
# # 9.80E-01,
# # 1.01E+00,
# # 1.08E+00,
# # 1.14E+00,
# # 1.15E+00,
# # 1.17E+00,
# # 1.22E+00,
# # 1.23E+00,
# # 1.35E+00,
# # 1.42E+00,
# # 1.42E+00,
# # 1.48E+00,
# # 1.74E+00,
# # 1.87E+00,
# # 1.49E+00,
# # 1.70E+00,
# # 1.64E+00,
# # 1.56E+00]

# # rem100Nb=[1.46E-01,
# # 6.17E-01,
# # 7.56E-01,
# # 8.48E-01,
# # 8.82E-01,
# # 9.81E-01,
# # 1.01E+00,
# # 1.08E+00,
# # 1.14E+00,
# # 1.14E+00,
# # 1.16E+00,
# # 1.19E+00,
# # 1.19E+00,
# # 1.28E+00,
# # 1.31E+00,
# # 1.29E+00,
# # 1.37E+00,
# # 1.53E+00,
# # 1.65E+00,
# # 1.57E+00,
# # 3.08E+00,
# # 1.64E+00,
# # 1.56E+00]

# """NEOS STUFF"""
# E=[2.00E+00,
# 2.20E+00,
# 2.40E+00,
# 2.60E+00,
# 2.80E+00,
# 3.00E+00,
# 3.20E+00,
# 3.40E+00,
# 3.60E+00,
# 3.80E+00,
# 4.00E+00,
# 4.20E+00,
# 4.40E+00,
# 4.60E+00,
# 4.80E+00,
# 5.00E+00,
# 5.20E+00,
# 5.40E+00,
# 5.60E+00,
# 5.80E+00,
# 6.00E+00,
# 6.20E+00,
# 6.40E+00,
# 6.60E+00,
# 6.80E+00,
# 7.00E+00,
# 7.20E+00,
# 7.40E+00,
# 7.60E+00]

# R=[4.57E-01,
# 7.07E-01,
# 7.90E-01,
# 8.34E-01,
# 8.73E-01,
# 8.95E-01,
# 9.23E-01,
# 9.83E-01,
# 1.00E+00,
# 9.87E-01,
# 1.02E+00,
# 1.07E+00,
# 1.08E+00,
# 1.10E+00,
# 1.09E+00,
# 1.06E+00,
# 1.07E+00,
# 1.10E+00,
# 1.09E+00,
# 1.13E+00,
# 1.19E+00,
# 1.23E+00,
# 1.20E+00,
# 1.18E+00,
# 1.24E+00,
# 1.27E+00,
# 1.31E+00,
# 1.34E+00,
# 1.39E+00]

# Err=[2.77E-02,
# 2.44E-02,
# 1.93E-02,
# 1.66E-02,
# 1.56E-02,
# 1.54E-02,
# 1.51E-02,
# 1.64E-02,
# 1.60E-02,
# 1.64E-02,
# 1.64E-02,
# 1.80E-02,
# 1.77E-02,
# 1.82E-02,
# 1.84E-02,
# 1.77E-02,
# 1.86E-02,
# 2.02E-02,
# 2.06E-02,
# 2.25E-02,
# 2.41E-02,
# 2.77E-02,
# 2.83E-02,
# 2.97E-02,
# 3.58E-02,
# 4.39E-02,
# 5.24E-02,
# 6.15E-02,
# 7.80E-02]

# Sum=[1.89E-01,
# 6.38E-01,
# 7.42E-01,
# 8.13E-01,
# 8.51E-01,
# 9.15E-01,
# 9.07E-01,
# 9.67E-01,
# 9.73E-01,
# 1.01E+00,
# 1.01E+00,
# 1.06E+00,
# 1.06E+00,
# 1.12E+00,
# 1.08E+00,
# 1.09E+00,
# 1.12E+00,
# 1.12E+00,
# 1.13E+00,
# 1.14E+00,
# 1.19E+00,
# 1.22E+00,
# 1.21E+00,
# 1.21E+00,
# 1.23E+00,
# 1.26E+00,
# 1.36E+00,
# 1.37E+00,
# 1.39E+00,
# 1.36E+00,
# 1.57E+00,
# 2.42E+00,
# 1.36E+00,
# 1.33E+00,
# 1.43E+00]

# rem4=[1.90E-01,
# 6.44E-01,
# 7.48E-01,
# 8.17E-01,
# 8.50E-01,
# 9.13E-01,
# 8.95E-01,
# 9.36E-01,
# 9.53E-01,
# 9.88E-01,
# 1.01E+00,
# 1.06E+00,
# 1.06E+00,
# 1.12E+00,
# 1.08E+00,
# 1.09E+00,
# 1.12E+00,
# 1.12E+00,
# 1.13E+00,
# 1.14E+00,
# 1.19E+00,
# 1.22E+00,
# 1.21E+00,
# 1.21E+00,
# 1.23E+00,
# 1.26E+00,
# 1.36E+00,
# 1.37E+00,
# 1.39E+00,
# 1.36E+00,
# 1.57E+00,
# 2.42E+00,
# 1.36E+00,
# 1.33E+00,
# 1.43E+00]

# plt.errorbar(E, R, Err, linestyle="", marker="s", elinewidth=2, alpha=1, label="Exp")
# plt.plot(E, Sum[:-6], alpha=.7, label="NEOS Sum")
# plt.plot(E, rem4[:-6], label="Removed 92Y, 99Zr, 99Nb, 143La", alpha=.7)
# #plt.plot(E, rem92Rb[:-1], label="Removed 92Rb", alpha=.7)
# plt.xlim((2, 7))
# plt.ylim((.6, 1.4))
# plt.xlabel("Energy [MeV]")
# plt.ylabel("S$_i$/S$_{i+1}$")
# plt.legend()
# plt.savefig("NEOSimp.svg", bbox_inches="tight")
# plt.show()

fig, axs=plt.subplots(2, 1, sharex=True, sharey=True, figsize=(13.5, 16))
fig.subplots_adjust(wspace=0, hspace=0)

calcUnc=[]
for i in range(len(NEOS.S)):
    calcUnc.append((1/RENO.S[i])*math.sqrt(NEOS.cov[i][i]+(NEOS.S[i]/RENO.S[i])**2*RENO.cov[i][i]))


axs[0].plot(NEOS.E_avg, np.array(sumNEOS.R), label="Summation")
axs[0].plot(NEOS.E_avg, np.array(hmNEOS.R), linestyle="--",label="Huber Mueller", color="#008176", alpha=.7)

axs[1].plot(RENO.E_avg, np.array(sumRENO.R), label="Summation")
axs[1].plot(RENO.E_avg, np.array(hmRENO.R),linestyle="--", label="Huber Mueller", color="#008176", alpha=.7)

axs[0].errorbar(NEOS.E_avg, NEOS.R, NEOS.delR[:-1], label="Exp.",  linestyle="", color= "#0000a7", marker="s", elinewidth=2, alpha=1)
axs[1].errorbar(RENO.E_avg, RENO.R, RENO.delR[:-1], label="Exp.", linestyle="", color="#0000a7", marker="s", elinewidth=2, alpha=1)

plt.xlim((2, 7.8))
plt.ylim((.6, 1.5))
axs[0].set(ylabel="NEOS")
axs[1].set(ylabel="RENO")
fig.supxlabel("Energy[MeV]")
#fig.supylabel("$S_i/S_{i+1}$")
plt.legend()
# plt.savefig("adjRat.svg", bbox_inches="tight")
plt.show()

# plt.plot(NEOS.E, NEOS.S, alpha=.8, label="NEOS")
# plt.plot(hmNEOS.E, hmNEOS.S, alpha=.8, label="Huber Mueller")
# plt.plot(sumNEOS.E, sumNEOS.S, alpha=.8, label="Summation")
# plt.xlabel("$\\overline{\\nu_e}$ Energy [MeV]")
# plt.ylabel("Spectra [MeV$^{-1}$]")