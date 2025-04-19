from collections.abc import Iterable
import re
import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
from numpy import array
import sys
import glob
import csv
from itertools import combinations
import mplcursors
#from array import array
from scipy.stats import norm
import random
from random import randrange, random, gauss, seed
import ast
np.set_printoptions(threshold=sys.maxsize)

#Ensures that excessively large csv files can be properly read
import sys
import csv
maxInt = sys.maxsize
while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)
#plt.style.use('custom')

class Util:
    """
    Utility class with methods (all static methods) to facilitate other analysis processes
    (No methods with the sole intention of graphing, those get relegated to graphUtil)
    
    Class Methods:
        MCSim: runs Monte Carlo simulation
        strToStrArr: reads out a string form of an array to an actual array (containing strings)
        readPlotCSV: reads out array containing plot data
        chiSq: calculates chi squared value for array
        dictToCSV: converts dictionary of spectra to a CSV file
        readDict: reads CSV file to a dictionary of spectra form
        getIndex: gets index of an energy in MeV relative to bins
        pickLarge: gets largest individual nuclides in contribution to spectra in a region
        pickCFY_Q: picks nuclides based on sufficiently large CFY and Q in region of interest
        gradDesc: calculates a gradient descent model to find nuclides whose removal best improves chi squared fit
        getExpData: outputs the experimentally collected data for the desired reactor source (NEOS, RENO, Daya Bay)
        getBin: outputs the correct bin, allowing for None-> standard bin in method parameters
    """ 
    
    @staticmethod
    def MCSim(pdfInf, numIter=10**6, **kwargs):
        """
        Simulates a Reactor Monte Carlo simulation, and can account for gaussian of detector resolution and rebinning
    
        Args:
            pdfInf (arraylike containing 2 elements): [bins, ___]
                ___ either represents pdf (in which case it is an arraylike of floats same length as bins)
                    OR it represents an already constructed tree, which can be used
            numIter (int, optional): number of observations to make. Defaults to 10**6
            **kwargs:
                -res: let this be a parameter equal to False if you DON'T want to run the gaussian resolution filter
                    By default, resolution is accounted for
                -bin(arraylike): let this be the new binnings, otherwise returns data unbinned 
                -promptS (float): if present, shifts energy to convert from antineutrino to prompt energy
                    Defaults to .78
                -resP (float): if present, changes the percentage of sqrt(promptEnergy) for gaussian standard deviation
                    Defaults to .03 (NEOS value)
                -raw(boolean): if present, indicates that data should be outputted in the animation format
                -tree (file name): if present, indicates a file to read preconstructed tree for cdf off of
        
        Returns:   
            sumSpectra object for calculated spectra
        """
        print("inMCSim")
        if "promptS" in kwargs:
            promptS=kwargs["promptS"]
        else:
            promptS=.78
        if "resP" in kwargs:
            resP=kwargs["resP"]
        else:
            resP=.03
        bins=pdfInf[0]
        if isinstance(pdfInf[1], str):
            tree=BSTree.treeFromFile(pdfInf[1])
        else:
            pdf=pdfInf[1]
            cdf=np.zeros(len(bins))
            for i in range(len(bins)):
                if len(cdf)==0:
                    continue
                cdf[i]=cdf[-1]+(bins[1]-bins[0])*pdf[i]
            tree=BSTree.makeTree(cdf)
        spect=np.zeros(len(bins))
        for x in range(numIter):
            i=random()
            ind=tree.search(i)
            E=bins[int(ind)]
            if "res" not in kwargs:
                Enew=E+(gauss()*resP*math.sqrt(E-promptS)) 
                if Enew>8: Enew=8
                if Enew<1.8: Enew=1.8
                Eround=round(Enew, int(-math.log10(bins[1]-bins[0])))
                index=int((Eround-1.8)*1/(bins[1]-bins[0]))
                if index==len(bins):
                    index=len(bins)-1
                spect[index]+=1
            else:
                spect[ind]+=1
            if x%1000==0:
                print("MCSim count:" +x)
        if "raw" in kwargs:
            if "bins" not in kwargs:
                return spect
        else:
            spect/=(numIter*(bins[1]-bins[0]))#To normalize, need to account for binwidth
        MC=sumSpectra({"E": bins, "S":spect}, ifSimp=True)#just transfers info to the data and doesn't calculate R (saves time)
        if "bins" in kwargs:
            MC.rebin(kwargs["bins"])
        if "raw" in kwargs:
            return MC.S
        MC.renorm()
        return MC
    
    @staticmethod
    def strToStrArr(data, delimeter, numCols):
        """
        Converts extended string of data to a 2D array with all entries in original string
            different rows lie on different lines in multiline string

        Args:
            data (string or array of strings): the string to be converted/read
            delimeter (string (char)): the dividing symbol between elements 
            numCols (int): number of columns in data

        Returns:
            list of lists: 2D array containing data
        """
        
        st=""
        if(isinstance(data, str)):
            st=data
        else:
            for x in data:
                st+=x+"\n"
        arr=[]
        for x in range(numCols):
            arr.append([])
        for line in st.splitlines():
            line=line.strip()
            fChar=list(re.finditer(re.escape(delimeter)+r'\S', line))
            bChar=list(re.finditer(r'\S'+re.escape(delimeter), line))
            if(len(fChar)<numCols-1):
                continue
            for x in range(numCols):
                frontInd=-1
                backInd=len(line)
                if x==0:
                    backInd=bChar[x].span()[0]
                elif x==numCols-1:
                    frontInd=fChar[-1].span()[0]
                else:
                    frontInd=fChar[x-1].span()[0]
                    backInd=bChar[x].span()[0]
                arr[x].append(line[frontInd+1:backInd+1]) #the front is shifted ahead by 1 since it starts on the delimeter space
        return arr                                        # the back is shifted ahead by one since the index starts on the last space then needs to get shifted to the delimeter

    @staticmethod
    def readPlotCSV(fileName):
        """
        Reads out the csv file of a digitized plot to 2 arrays
        
        Args:
            fileName(string): file name of csv
            
        Returns
            (arraylike of floats, arraylike of floats): x, y values in plot, respectively
        """
        with open(fileName, "r") as file:
            data=file.read()
        arr=[]
        for line in data.splitlines():
            arr.append(line)
        strArr=Util.strToStrArr(arr, ", ", numCols=2)
        x, y=np.zeros(len(strArr[0])), np.zeros(len(strArr[0]))
        for i in range(len(strArr[0])):
            x[i]=float(strArr[0][i])
            y[i]=float(strArr[1][i])
        return x, y

    @staticmethod
    def chiSq(arrExp, arrTest, ifLS=False, lims=[1.8, 8], bins=None, ifDoubSq=False):
        """Determines the chi-square and chi-square per point for the data

        Args:
            arrExp (array of floats): Expected array (i.e. models)
            arrTest (array of floats): the experimental data
                NOTE: arrExp and arrTest MUST have the same length
            ifLS (boolean, optional): if True, just computes a least-squares calculation. Defaults to False
            lims (arraylike of 2 floats, optional): the limits over which Chi squared should be added. Defaults to [1.8, 8]
            bins (arraylike of floats, optional): bins to be used. Defaults to Util.getBin default
            
        Returns:
            (float, float): total chi-square value, average chi-square per point
        """
        if isinstance(bins, type(None)):
            inds=[0, len(arrExp)]
        else:
            inds=Util.getIndex(lims, bins)
        chiSq=0
        for x in range(len(arrExp)):
            if ifLS:
                chiSq+=(arrExp[x]-arrTest[x])**2                
            else:
                chiSq+=(arrExp[x]-arrTest[x])**2/arrExp[x]
        return chiSq, chiSq/len(arrExp)

    @staticmethod
    def dictToCSV(endFileName, dict):
        """
        Converts folder of nuclide data into a CSV database of dictionaries
        
        Parameters:
            endFileName (string): the name for the csv file created
            dict (dictionary of Spectra objects OR dictionary of dictionaries referring to object): spectra to be converted to CSV
        
        Returns:
            Nothing (but saves the csv file in location specified)
        """
        dTrans={}
        for x in dict:
            if isinstance(dict[x], type({"a":5})):
                dTrans[x]=dict[x]
            else:
                dTrans[x]=dict[x].__dict__
        with open(endFileName, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in dTrans.items():
                writer.writerow([key, value])
    
    @staticmethod
    def readDict(spectraType, filePath, saveFile=None):
        """
        Reads csv file from analyzed dictionary and converts to dictionary of spectra

        Args:
            spectraType (Spectra type): the Spectra type that the csv objects belong to
            filePath (string): file path to access csv file.
            saveFile (string, optional): If string present, saves all sources as files written out. Defaults to no save

        Returns:
            (dictionary of nuclideSpectra): contains all spectra in csv, with the key being their name
        """
        dict={}
        with open(filePath, mode ='r')as file:
        # reading the CSV file
            csvFile = csv.reader(file)
        # displaying the contents of the CSV file
            count=0
            for lines in csvFile:
                count+=1
                if len(lines)==2:
                    dict[lines[0]]=spectraType(eval(lines[1]))
                    if not isinstance(saveFile, type(None)):
                        with open(saveFile+lines[0]+".csv", "w") as file:
                             writer=csv.writer(file)
                             for key, value in dict[lines[0]].__dict__.items():
                                 writer.writerow([key, value])
        return dict

    @staticmethod
    def getIndex(ranE, bins=None):
        """
        Outputs bin indices from a continuous energy range

        Args:
            ranE (arraylike of 2 floats): in the form (E_min, E_max)
            bins (arraylike of floats, optional): the bin edges used. Defaults to Util.getBin default
            
        Returns:
            (2 float array): [lowInd, highInd]
        """
        bins=Util.getBin(bins)
        Ebds=[0, 0]
        for x in range(len(bins)-1):
            if (bins[x]<=ranE[0])&(bins[x+1]>ranE[0]):
                Ebds[0]=x
            if (bins[x]>ranE[1]):
                Ebds[1]=x 
                break   
        if Ebds[1]==0:
            Ebds[1]=-1
        return Ebds
    
    @staticmethod 
    def pickLarge(dict, ranE, bins, num=5):
        """
        Outputs a number of nuclides with largest magnitudes in a certain interval
                
        Args:
            dict (dictionary of spectra): the nuclides from which the selection is made
            ranE (arraylike of 2 floats): the energy range of interest
            bins (arraylike of floats): the bins to be used for the data
            num (int, optional): number of nuclides to be picked. Defaults to 5.
        
        Returns:
            (list of strings): the nuclides selected, in standard nomenclature
        """

        Ebds=Util.getIndex(ranE, bins=bins)
        eLev=[]
        names=[]
        largest=[]
        for x in dict:
            try:
                eLev.append(np.nanmax(np.array(dict[x].S[Ebds[0]:Ebds[1]])))
                names.append(x)
            except RuntimeWarning:
                pass
        for i in range(num):
            ind=np.nanargmax(np.array(eLev))
            largest.append(names[ind])
            eLev=np.delete(eLev, ind)
            names=np.delete(names, ind)
        return largest
        
    @staticmethod
    def pickCFY_Q(fisFrac, ranE, capCFY=.01):
        """
        Selects sources based on sufficiently large CFY (>0.01) and the Q value being ~ranE
        
        Args:
            fisFrac (dictionary of 4 floats): contains the reactor fission fractions for u5, u8, pu9, and pu1
            ranE (arraylike of 2 floats): [lowE, highE] for feature of interest
            capCFY (float, optional): min CFY percentage used as filter. Defaults to .01
            
        Returns:
            (array of strings): names of the nuclides that pass the Q and CFY filters
        """       
        CFY={}
        for nucl in ["u5", 'u8', 'pu9', 'pu1']:
            fileName="Nuclide Spectral Data\\" +nucl+"cfy.txt"
            with open(fileName, "r") as file:
                data=file.read()
            strArr=Util.strToStrArr(data, delimeter=";", numCols=4)
            for i in range(len(strArr[0])):
                strKey="z_"+str(strArr[0][i])+"_a_"+str(strArr[1][i])+"_liso_"+str(strArr[2][i])
                if strKey not in nuclideSpectra.names:
                    strKey+="_t"
                if nucl=="u5":
                    CFY[strKey]=float(strArr[3][i])*fisFrac[nucl]
                else:
                    CFY[strKey]+=float(strArr[3][i])*fisFrac[nucl]   
        remArr=[]
        for nucl in CFY:
            if CFY[nucl]<capCFY:
                remArr.append(nucl)
        for nucl in remArr:
            del CFY[nucl]
        with open("READ FROM CODE\\nndc_nudat_data_export(1).csv", "r") as file:
            data=file.read()
        arr=[]
        for line in data.splitlines():
            arr.append(line)
        z, a, Q=[], [], []
        strArr=Util.strToStrArr(arr[1:], ",", 3)
        for i in range(len(strArr[1])):
            z.append(float(strArr[1][i]))
            a.append(float(strArr[0][i])+float(strArr[1][i]))
            Q.append(float(strArr[2][i])/1000)
        fNucl=[]
        for nucl in CFY:
            und=list(re.finditer(r"_", nucl))
            zNucl=int(nucl[und[0].span()[0]+1: und[1].span()[0]])
            aNucl=int(nucl[und[2].span()[0]+1: und[3].span()[0]])
            for i in range(len(z)):
                if (z[i]==zNucl) and (a[i]==aNucl):
                    if (ranE[0]<=Q[i]) and (Q[i]<=ranE[1]):
                        fNucl.append(nucl)
                    break
        return fNucl
    
    @staticmethod
    def gradDesc(aType, ranE,numReps=5, depth=1, sizeDict=150, ifPlot=True,
                 bins=None, ranGraph=None, source="NEOS"):
        """
        Outputs a certain number of isotopes that have the greatest contribution to reducing sum of squares in a specific energy region
        
        Args:
            aType (String): either "S", "R", or "MD" to indicate analysis form
                ~S: Spectra data
                ~R: Spectral ratios
                ~MD: Model Data. calculates the ratio between the model and the data in a specific region and compares the distance between that and unity (which is the value of the plot given that the experimental data and model agree)
            ranE (arraylike of 2 floats): in the form [eMin, eMax]
            numReps (int, optional): number of times that the search loop is repeated
                Defaults to 5
            depth (int, optional): amount of nuclides to include in each repetition for optimality
                Run time factorially increases with the depth. Defaults to 1
            sizeDict (int, optional): the max desired number of nuclides to be chosen from 
                Lower values increase run speed since less combinations. Defaults to 150
                Will choose from the most contributing nuclides in the chosen energy range
            ifPlot (boolean, optional): if true, plots the change as each nuclide is removed   
                as well as including the difference of squares value. Defaults to True
            bins (array of floats, optional): the binning to which the dictionary (and the raw data) 
                will be binned to. Defaults to Util.getBin default
            ranGraph (arraylike of 2 floats, optional): the bounds for the graph itself. 
                Defaults to None, indicating to use the same bounds as ranE
            
        Returns:
            (array of strings): contains the names of the removed nuclides
            """
        bin=Util.getBin(bins)
        graphBds=ranE
        if not isinstance(ranGraph, graphBds):
            graphBds=ranGraph
        data, dict=Util.getExpData(source)
        #totDict=sumSpectra.binDict(dict, bin)
        data.rebin(bin)
        data.renorm()
        disc=[]
        for x in dict:
            for i in range(len(totDict[x].disc)):
                if (ranE[0]<=dict[x].disc[i][0]<=ranE[1]):
                    disc.append(x)
                    break
        pickLarge=Util.pickLarge(ranE, dict, bins, num=sizeDict)
        remDict={}
        for key in disc:
            remDict[key]=totDict[key]
        for key in pickLarge:
            remDict[key]=totDict[key]
        
        #TESTING ZONE
        
        # del dict['z_37_a_88_liso_0']
        # del dict['z_40_a_101_liso_0']
        # del dict['z_39_a_97_liso_1']
        # del dict['z_38_a_97_liso_0']
        # del dict['z_39_a_99_liso_0']
        # del dict['z_41_a_103_liso_0']
        # del dict['z_57_a_147_liso_0']
        
        eInd=Util.getIndex(ranE, bins=bins)
        nucl=[]
        delName=[]
        for x in remDict:
            if remDict[x].epE<ranE[0]-.5:
                delName.append(x)
        for x in delName:
            del remDict[x]
        if aType=="MD":
            inte={}
            expS=np.array(data.S)                
        totalSq=0
        totalSp=sumSpectra.sumNucl(totDict, bins)
        totalSp.renorm()
        match aType:
            case "R":
                arrTest=totalSp.R
                arrExp=data.R
            case "S":
                arrTest=totalSp.S
                arrExp=data.S
            case "MD":
                arrTest=totalSp.S/expS
                arrExp=np.ones(len(arrTest))
        totalSq, _=Util.chiSq(arrExp, arrTest)
        if ifPlot:
            minY, maxY=[], []
            plt.xlim(graphBds)
            plt.xlabel("Energy [MeV]")
            plt.title("Sum Fine Struc in range "+str(ranE[0])+"-"+str(ranE[1])+" MeV: "+aType)
            match aType:
                case "S":
                    plt.plot(data.E, data.S, label="ExpData", color="k")
                    arr1=graphUtil.getYBounds(data.S, graphBds, bins=bins)
                    plt.plot(totalSp.E, totalSp.S, label="Total Sum $\chi^2:$ "+str(totalSq))
                    arr2=graphUtil.getYBounds(totalSp.S, graphBds, bins=bins)
                    plt.ylabel("Spectra [$MeV^{-1}$]")
                    minY.append(arr2[0])
                    maxY.append(arr2[1])
                case "R":
                    plt.plot(data.E_avg, data.R, label="ExpData", color="k")
                    arr1=graphUtil.getYBounds(data.R, graphBds, bins=bins)
                    plt.plot(totalSp.E_avg, totalSp.R, label="Total Sum $\chi^2$: "+str(totalSq))
                    arr2=graphUtil.getYBounds(data.S, graphBds, bins=bins)
                    plt.ylabel("$S_i/S_{i+1}$")
                    minY.append(arr2[0])
                    maxY.append(arr2[1])
                case "MD":
                    ratMD=totalSp.S/expS
                    plt.plot(data.E, ratMD, label="Total Sum $\chi^2$: "+str(totalSq))
                    plt.plot([0, 15], [1, 1], color="k", linewidth=4)
                    arr1=graphUtil.getYBounds(ratMD, graphBds, bins=bins)
                    plt.ylabel("Model/Data")
            minY.append(arr1[0])
            maxY.append(arr1[1])
        #totDict, inte=sumSpectra.renormDict(totDict, bins)
        for i in range(numReps):
            currElem=""
            count=0
            minSq=totalSq
            combs=combinations(remDict.keys(), depth)
            integScale=totalSp.renorm()
            norm=1
            totDict, inte=sumSpectra.renormDict(totDict, bins)
            for comb in combs:
                if count%1000==0:
                    print("Grad Desc Count: "+count)
                count+=1
                arrS=totalSp.S.copy()
                for x in comb:
                    temp=totDict[x].S
                    temp[np.isnan(temp)]=0
                    arrS-=temp
                match aType:
                    case "S":
                        norm=1
                        for x in comb:
                            norm-=inte[x]
                        arrS/=norm  #NOTE: SOMETHING REALLY WEIRD IS HAPPENING WITH THIS NORM
                        arrTest=arrS
                    case "R":
                        arrTest=np.zeros(len(totalSp.E)-1)
                        for j in range(len(totalSp.E)-1):
                            if (arrS[j+1]!=0):
                                arrTest[j]=arrS[j]/arrS[j+1]
                            else:
                                arrTest[j]=arrS[j]/1E-20
                    case "MD":
                        norm=1
                        for x in comb:
                            norm-=inte[x]
                        arrS/=norm  #NOTE: SOMETHING REALLY WEIRD IS HAPPENING WITH THIS NORM
                        arrTest=arrS/expS
                sq, _=Util.chiSq(arrExp, arrTest, lims=ranE, bins=bins)
                if (sq<minSq):
                    currComb=comb
                    minSq=sq
            if minSq<totalSq:
                nucl.append(currComb)
                for isot in currComb:
                    del remDict[isot]
                    del totDict[isot]
                totalSq=minSq
                totalSp=sumSpectra.sumNucl(totDict, bins)
                totalSp.renorm()

                if ifPlot:
                    match aType:
                        case "S":
                            plt.plot(totalSp.E, totalSp.S, label="Step "+str(i+1)+" $\chi^2$: "+str(minSq))
                            arr=graphUtil.getYBounds(totalSp.S, graphBds, bins=bins)
                        case "R":
                            plt.plot(totalSp.E_avg, totalSp.R, label="Step "+str(i+1)+" $\chi^2$: "+str(minSq))
                            arr=graphUtil.getYBounds(totalSp.R, graphBds, bins=bins)    
                        case "MD":
                            ratMD=totalSp.S/expS
                            plt.plot(totalSp.E, ratMD, label="Step "+str(i+1)+"$\chi^2$: "+str(minSq))
                            arr=graphUtil.getYBounds(ratMD, graphBds, bins=bins)
                    minY.append(arr[0])
                    maxY.append(arr[1])
            else:
                break
        if ifPlot:
            plt.legend()
            saveName="SumImprove 6_29\\SumFineStruc "+str(ranE[0])+"-"+str(ranE[1])+" MeV "+ aType +".png"
            plt.savefig(saveName)
            plt.show()
            plt.ylim([min(minY), max(maxY)])
            mplcursors.cursor()
            
        return nucl                       

    @staticmethod
    def getExpData(source, bins=None):
        """
        Outputs NEOS, RENO, or Daya Bay data and summation dictionary depending on source value input
        
        Args:
            source(string): the string associated with desired data. Possible inputs:NEOS, RENO, Daya50, Daya250 (for 50 KeV or 250 KeV binned data)
            bins (arraylike of floats, optional): binning to put expData into. Defaults to Util.getBin default
            
        Returns:
            (redRelSpectra, dictionary of nuclideSpectra): object for data, dictionary of its associated summation
        """
        data=5
        match source:
            case ("Daya50"|"Daya250"):
                analyzedSpectra=Util.readDict(redRelSpectra, filePath="READ FROM CODE\\DayaBay Analyzed.csv")
                if "250" in source:
                    data=analyzedSpectra["Daya250"]
                    data.E=dayaBay250.bins[:-1]
                    # for i in range(len(data.E)):
                        
                    #     Util.getIndex((data.E[i]-.125, data.E[i]+.125), bins=np.arange(0, 10, .01))
                else:
                    data=analyzedSpectra["Daya50"]
                    data.E=np.array(data.E)+.78
                fileName="READ FROM CODE\\indivNuclidesDAYA.csv"
            case ("NEOS"|"RENO"):
                analyzedSpectra=Util.readDict(redRelSpectra, filePath="READ FROM CODE\\NEOS+RENO analyzedData.csv")
                NEOS, RENO=analyzedSpectra["NEOS"], analyzedSpectra["RENO"]
                if source=="NEOS":
                    data=NEOS
                    fileName="READ FROM CODE\\indivNuclidesNEOS.csv"
                else:
                    data=RENO
                    fileName="READ FROM CODE\\indivNuclidesRENO.csv"
        if not isinstance(bins, type(None)):
            data.rebin(bins)
        if isinstance(data.S, type([5])):
            data.S=np.array(data.S)
        data.renorm()
        dict=Util.readDict(nuclideSpectra, filePath=fileName)
        return data, sumSpectra.renormDict(dict, np.arange(1.8, 10, .01))[0]
 
    @staticmethod
    def getBin(bins):
        """Outputs the correct binning; defaults to np.arange(1.8, 8.2, .2)"""
        if isinstance(bins, type(None)):
            return np.arange(1.8, 8.2, .2)
        return bins
        
class graphUtil:
    """
    Contains the utility methods specifically interested in graphing/graphs
    
    Class Methods:
        graphRatio: plots adjacent ratio plot
        getYBounds: obtains appropriate y-bounds for a given function/graph
        plotSpectComb: plots combinations of removing sources from a summation model, in relation to adjacent ratio plot
        graphGrid (DEPRECATED): graphs 4 grid including spagetti plot, ratio plot, and maybe residuals
    """

    @staticmethod    
    def graphRatio(ranE, remNucl=None, spag=None, ifLeg=False, ifHover=False, bins=None, source="NEOS"):
        """Creates a graph comparing the experimental data (Daya Bay) and
        some subset of the total collection of arrays for the spectral ratios

        Args:
            ranE (arraylike with 2 floats): of the form (Emin, Emax)
            remNucl (arraylike of strings, optional): list of the nuclides to be removed for test sum. 
                Defaults to none
            spag (arraylike of floats, optional): if given, plots spagetti plot to the left with the specific
                sources indicted. Defaults to no plot
            ifLeg (bool, optional): if true, plots legend on spagetti plot. Defaults to True
            ifHover (bool, optional): if true, only hovering over a plot shows annotation; 
                otherwise requires a click. Defaults to False
            bins (arraylike of floats, optional): the bins to plot to. Defaults to np.arange(1.8, 8, .2)
            source (string, optional): indicates the reactor on which the reactor data is needed 
                Options: "NEOS", "RENO", "Daya". Defaults to "NEOS"
            
        Returns:
            None (but prints out the graphs)
        """
        bins=Util.getBin(bins)
        data, dict=Util.getExpData(source, bins)
        if not isinstance(remNucl, type(None)):
            dict2=dict.copy()
            for x in remNucl:
                del dict2[x]
            sumSp=sumSpectra.sumNucl(dict2, bins=bins)
        totSum=sumSpectra.wholeSum()
        if not isinstance(spag, type(None)):
            spagBds=[0,0]
            fig, axs=plt.subplots(1, 2, sharex=True)
            for x in dict:
                if x in spag:
                    axs[0].plot(dict[x].E, dict[x].S, label=x, alpha=1)
                    tempBds=graphUtil.getYBounds(dict[x].S, ranE, bins=bins)
                    if tempBds[0]<spagBds[0]:
                        spagBds[0]=tempBds[0]
                    if tempBds[1]>spagBds[1]:
                        spagBds[1]=tempBds[1]
            axs[0].set_ylim(spagBds)
            if ifLeg:
                axs[0].legend()
            axs[0].set(xlabel="Energy [MeV]", ylabel="Spectrum $[MeV^{-1}]$")
            axs[1].scatter(data.E_avg, data.R, color="g", label="Exp", alpha=.75)
            axs[1].scatter(totSum.E_avg, totSum.R, color="b", label="Total Sum", alpha=.75)
            if not isinstance(remNucl, type(None)):
                axs[1].scatter(sumSp.E_avg, sumSp.R, color="r", label="Test Sum", alpha=.75)
                sumBds=graphUtil.getYBounds(sumSp.R, ranE, bins=bins)
            else:
                sumBds=[sys.maxint, sys.minint]
            axs[1].set_xlim(ranE)
            dataBds=graphUtil.getYBounds(data.R, ranE, bins=bins)
            totBds=graphUtil.getYBounds(totSum.R, ranE, bins=bins)
            axs[1].set_ylim((min(dataBds[0], totBds[0], sumBds[0]), max(dataBds[1], totBds[1], sumBds[1])))
            axs[1].set(xlabel="Energy [MeV]", ylabel="$S_i/S_{i+1}$")
            axs[1].legend()
        else:
            fig, ax=plt.subplots()
            ax.scatter(data.E_avg, data.R, color="g", label="Exp", alpha=.75)
            ax.scatter(totSum.E_avg, totSum.R, color="b", label="Total Sum", alpha=.75)
            if not isinstance(remNucl, type(None)):
                ax.scatter(sumSp.E_avg, sumSp.R, color="r", label="Test Sum", alpha=.75)
                sumBds=graphUtil.getYBounds(sumSp.R, ranE, bins=bins)
            else:
                sumBds=[sys.maxint, sys.minint]
            ax.set_xlim(ranE)
            dataBds=graphUtil.getYBounds(data.R, ranE, bins=bins)
            totBds=graphUtil.getYBounds(totSum.R, ranE, bins=bins)
            axs.set_ylim((min(dataBds[0], totBds[0], sumBds[0]), max(dataBds[1], totBds[1], sumBds[1])))
            ax.set(xlabel="Energy [MeV]", ylabel="$S_i/S_{i+1}$")
            ax.legend()
        mplcursors.cursor(hover=ifHover)
        plt.show()
            
    @staticmethod
    def getYBounds(yData, ranE, margin=.1, bins=None):
        """Determines an upper and lower bound for the plot given the y data presented

        Args:
            yData (array): array containing the data, binned in accordance with bins
            ranE (arraylike w/ 2 floats): low and high energy bound for plot
            margin (float, optional): width of margin, in percentage between highest and lowest
            y values. Defaults to .1    
            bins (arraylike with floats, optional): the bins used. Defaults to Util.getBin default
        
        Returns:
            tuple of 2 floats: (bottom y bound, top y bound)  
        """
        bin=Util.getBin(bins)
        Ebds=Util.getIndex(ranE, bins=bin)
        yDisp=np.array(yData[Ebds[0]: Ebds[1]])
        d=np.max(yDisp)-np.min(yDisp)
        top=np.max(yDisp)+d*margin
        bot=np.min(yDisp)-d*margin
        return (bot, top)        
    
    @staticmethod    
    def plotSpectComb(isot, ranE, form="add", ifLeg=True, remNucl=None, ifSpag=False, bins=None, source="NEOS", save=None):
        """Plots adjacent spectral ratio plot, changing presence/lack of a few isotopes

        Args:
            isot (array of strings): names of the isotopes to be removed
            ranE (arraylike of 2 floats): of the form (E_min, E_max)
            form (string): either 'add' or 'sub'
                -if add: the plot will start with all the nuclides removed and adds one at a time from the isot list
                -if sub: the plot will be constructed by subtracting 1 nuclide at a time from the total sample
            ifLeg (bool, optional): if True, shows a legend. Defaults to True
            remNucl(arraylike of strings, optional): Nuclides to remove from the dictionary. Defaults to None 
                If none, uses whole dictionary. Defaults to None
            ifSpag (bool, optional): if True, displays spagetti plot to the left. Defaults to False
            bins (arraylike of floats): the binning to use. Defaults to Util.getBin default
            source (string, optional): indicates the reactor on which the reactor data is needed 
                Options: "NEOS", "RENO", "Daya". Defaults to "NEOS"
            save (string, optional): indicates where to save the file to
            
        Returns:
            None (but prints out graphs)
        """
        data, dict=Util.getExpData(source, bins)
        if not isinstance(remNucl, type(None)):
            for x in remNucl:
                del dict[x]        
        sumSpectra.renormDict(dict, bins=np.arange(1.8, 10, .01))
        bins=Util.getBin(bins)
        wholeSum= sumSpectra.sumNucl(dict, bins=bins)
        if ifSpag:
            fig, axs=plt.subplots(1, 2, sharex=True)
            for x in isot:
                axs[0].plot(dict[x].E, dict[x].S, label=x, alpha=1)
            axs[0].set(xlabel="Energy [MeV]", ylabel="Spectra [$MeV^{-1}$]")
            axs[1].plot(data.E_avg, data.R, label="Exp", alpha=.75)
            EBds=graphUtil.getYBounds(data.R, ranE, bins=bins)
            yBds=[EBds[0], EBds[1]]
            temp=dict.copy()
            for x in isot:
                string=""
                if form=="sub":
                    if x in dict:
                        del dict[x]
                    string= "No "
                elif form=="add":
                    string="Only "
                    for y in isot:
                        if (x!=y) & (y in dict):
                            del dict[y]
                sumSp=sumSpectra.sumNucl(dict, bins=bins)
                axs[1].plot(sumSp.E_avg[:-2], sumSp.R[:-2], label=string+x, alpha=.75)
                dict=temp.copy()
                Ebds=graphUtil.getYBounds(sumSp.R[:-2], ranE, bins=bins)
                if (Ebds[0]<yBds[0]):
                    yBds[0]=Ebds[0]
                if (Ebds[1]>yBds[1]):
                    yBds[1]=Ebds[1]
            for x in isot:
                if x in dict:
                    del dict[x]
            sumSp=sumSpectra.sumNucl(dict, bins=bins)
            axs[1].plot(sumSp.E_avg[:-2], sumSp.R[:-2], label="All rem.", alpha=.75)
            Ebds=graphUtil.getYBounds(sumSp.R[:-2], ranE, bins=bins)
            if (Ebds[0]<yBds[0]):
                yBds[0]=Ebds[0]
            if (Ebds[1]>yBds[1]):
                yBds[1]=Ebds[1]
            axs[1].plot(wholeSum.E_avg[:-2], wholeSum.R[:-2], label="Full Sum", alpha=1, color="k")
            Ebds=graphUtil.getYBounds(wholeSum.R[:-2], ranE, bins=bins)
            if ifLeg:
                axs[1].legend()
            if (Ebds[0]<yBds[0]):
                yBds[0]=Ebds[0]
            if (Ebds[1]>yBds[1]):
                yBds[1]=Ebds[1]
            axs[1].set_xlim(ranE)
            axs[1].set_title("Spectral ratios")
            axs[1].set(xlabel= "Energy [MeV]", ylabel="$S_i/S_{i+1}$")
            axs[1].set_ylim(yBds)
        else:
            plt.errorbar(data.E_avg, data.R,data.delR[:-1], label="Exp", marker="o", linestyle="", alpha=1, elinewidth=2)
            EBds=graphUtil.getYBounds(data.R[:-2], ranE, bins=bins)
            yBds=[EBds[0], EBds[1]]
            temp=dict.copy()
            for x in isot:
                string=""
                if form=="sub":
                    if x in dict:
                        del dict[x]
                    string= "No "
                elif form=="add":
                    string="Only "
                    for y in isot:
                        if (x!=y) & (y in dict):
                            del dict[y]
                sumSp=sumSpectra.sumNucl(dict, bins=bins)
                plt.plot(sumSp.E_avg, sumSp.R, label=string+x, alpha=.75)
                dict=temp.copy()
                Ebds=graphUtil.getYBounds(sumSp.R[:-2], ranE, bins=bins)
                if (Ebds[0]<yBds[0]):
                    yBds[0]=Ebds[0]
                if (Ebds[1]>yBds[1]):
                    yBds[1]=Ebds[1]
            for x in isot:
                if x in dict:
                    del dict[x]
            sumSp=sumSpectra.sumNucl(dict, bins=bins)
            plt.plot(sumSp.E_avg[:-2], sumSp.R[:-2], label="All rem.", alpha=.75)
            Ebds=graphUtil.getYBounds(sumSp.R[:-2], ranE, bins=bins)
            if (Ebds[0]<yBds[0]):
                yBds[0]=Ebds[0]
            if (Ebds[1]>yBds[1]):
                yBds[1]=Ebds[1]
            plt.plot(wholeSum.E_avg[:-2], wholeSum.R[:-2], label="Full Sum", alpha=1, color="k")
            Ebds=graphUtil.getYBounds(wholeSum.R[:-2], ranE, bins=bins)
            if (Ebds[0]<yBds[0]):
                yBds[0]=Ebds[0]
            if (Ebds[1]>yBds[1]):
                yBds[1]=Ebds[1]
            plt.xlim(ranE)
            plt.title("Spectral ratios")
            plt.xlabel("Energy [MeV]")
            plt.ylabel("$S_i/S_{i+1}$")
            plt.ylim(yBds)
            if ifLeg:
                plt.legend()
        mplcursors.cursor()
        if not isinstance(save, type(None)):
            plt.savefig(save, bbox_inches="tight")
        plt.show()

    """DEPRECATED"""
    @staticmethod
    def graphGrid(spectDict, ranE, remNucl=None, ifResid=True, ifDoubResid=False, bins=None, source="NEOS"):
        """Creates a graph array comparing the experimental data  and
        some subset of the total collection of arrays, including graphs for both
        the spectrum itself and the spectral ratios

        Args:
            spectDict (dictionary of Spectra): dictionary of the spectra being used for this plot
            ranE (arraylike containing 2 floats)): of the form (Emin, Emax)
            ifResid (bool, optional): if True, graphs residual plots for both spectral ratio. Defaults to True.
            ifDoubResid (bool, optional): if True, graphs the difference between the total and
                exp dataset on residual plot. Defaults to False
            source (string, optional): indicates the reactor on which the reactor data is needed 
                Options: "NEOS", "RENO", "Daya". Defaults to "NEOS"
            
        Returns:
            None (but prints out the graphs)
        """
        bins=Util.getBin(bins)
        data, dict=Util.getExpData(source, bins)
        totSum=sumSpectra.sumNucl(dict=dict, bins=bins)
        if not isinstance(remNucl, type(None)):
            for x in remNucl:
                del dict[x]
            sumSp=sumSpectra.sumNucl(spectDict, bins=bins)
        h=1
        if ifResid:
            h=2
        fig, axs=plt.subplots(2, h, sharex=True)
        axs[0, 0].set_xlim(ranE)
        if ifResid:
            axs[0, 0].scatter(data.E, data.S,  label="Exp", alpha=.5)
            axs[0, 0].scatter(totSum.E, totSum.S,  label="Total Sum", alpha=.5)
            axs[0, 1].scatter(data.E_avg, data.R,  label="Exp", alpha=.5)
            axs[0, 1].scatter(totSum.E_avg, totSum.R,  label="Total Sum", alpha=.5)
            if not isinstance(remNucl, type(None)):
                axs[0, 0].scatter(sumSp.E, sumSp.S, label="Test Sum", alpha=.5)
                axs[0, 1].scatter(sumSp.E_avg, sumSp.R,  label="Test Sum", alpha=.5)
            axs[0, 0].plot([0, 10], [0, 0], color="k", linewidth=1)
            axs[0, 0].set(xlabel="Energy [MeV]", ylabel="Spectra [$MeV^{-1}$]")
            
            if ifDoubResid:
                axs[0, 0].scatter([], [], color="k", label="Difference in |Residual| (sum-exp)")
            fig.legend()
            
            axs[0, 1].set(xlabel="Energy [MeV]", ylabel="$S_i/S_{i+1}$")

            residSW=totSum.S-data.S
            residRW=totSum.R-data.R
            residST=sumSp.S-data.S
            residRT=sumSp.R-data.R
            
            if ifDoubResid:
                absA=np.vectorize(abs)
                dResidS=absA(residSW)-absA(residST)
                dResidR=absA(residRW)-absA(residRT)
            
            axs[1, 0].scatter(data.E, residSW, color="b", label="Total Sum", alpha=.5)
            axs[1, 0].scatter(data.E, residST, color="r", label="Test Sum", alpha=.5)
            axs[1, 0].plot([0, 10], [0, 0], color="k", linewidth=1)
            axs[1, 0].set(xlabel="Energy [MeV]", ylabel="Residual of Spectra [$MeV^{-1}$]")
            if ifDoubResid:
                axs[1, 0].scatter(data.E, dResidS, color="k", label="Double Residual", alpha=.5)

            axs[1, 1].scatter(data.E_avg, residRW, color="b", label="Total Sum", alpha=.5)
            axs[1, 1].scatter(data.E_avg, residRT, color="r", label="Test Sum", alpha=.5)
            axs[1, 1].plot([0, 10], [0, 0], color="k", linewidth=1)
            axs[1, 1].set(xlabel="Energy [MeV]", ylabel="Residual of $S_i/S_{i+1}$")
            if ifDoubResid:
                axs[1, 1].scatter(data.E, dResidS, color="k", label="Double Residual", alpha=.5)
        else:  
            axs[0].scatter(data.E, data.S, color="g", label="Exp", alpha=.5)
            axs[0].scatter(totSum.E, totSum.S, color="b", label="Total Sum", alpha=.5)
            axs[1].scatter(data.E_avg, data.R, color="g", label="Exp", alpha=.5)
            axs[1].scatter(totSum.E_avg, totSum.R, color="b", label="Total Sum", alpha=.5)
            if isinstance(remNucl, type(None)):
                axs[0].scatter(sumSp.E, sumSp.S, color="r", label="Test Sum", alpha=.5)
                axs[1].scatter(sumSp.E_avg, sumSp.R, color="r", label="Test Sum", alpha=.5)            
            axs[0].plot([0, 10], [0, 0], color="k", linewidth=1)
            axs[0].set(xlabel="Energy [MeV]", ylabel="Spectra [$MeV^{-1}$]")
            axs[0].set_xlim(ranE)
            plt.gca().legend()
            axs[1].set(xlabel="Energy [MeV]", ylabel="$S_i/S_{i+1}$")
        plt.show()

class Spectra:
    """
    General parent class for all spectra type (ignoring dayaBay250 (deprecated))
    
    Attributes:
        E (arraylike of floats): Energy values (in MeV)
        S (arraylike of floats): spectra values
        E_avg (arraylike of floats): midpoint of energy bin, len(E_avg)=len(E)-1
        R (arraylike of floats): S_i/S_{i+1}, len(R)=len(E_avg)
        binEdge (arraylike of floats): edges of current binning
        
        *Preferably arraylike of floats are np.array to get maximal functionality
        
    Methods:
        rebin: rebins E, S, R
        integ: returns integral over spectra
        renorm: renormalizes S to integral 1
        getBinInd: calculates indexes of new binning E values with regards to current binning
        changeFissYield: converts Spectra to different reactor fission yields by Taylor expanding Huber-Mueller model 
    """
    
    def __init__(self, dict, ifSimp=False):
        """Constructor for Spectra

        Args:
            dict (dictionary): at minimum contains E, S.
            ifSimp (bool, optional): If True, doesn't calculate R or get BinEdge. Defaults to False.
        """
        self.E=dict["E"]
        self.S=dict["S"]
        self.E_avg=np.zeros(len(self.E)-1)
        self.R=np.zeros(len(self.E)-1)
        if not ifSimp:
            if "R" in dict:
                self.E_avg=dict["E_avg"]
                self.R=dict["R"]
            else:
                self.calcR()
            if "binEdge" in dict:
                self.binEdge=dict["binEdge"]
            else:
                self.binEdge=HuberMueller.bins[int(10*round(self.E[2]-self.E[1], 1)-1)]
    
    def calcR(self):
        """Calculates E_avg and R values"""
        self.E_avg, self.R=np.zeros(len(self.E)-1), np.zeros(len(self.E)-1)
        for x in range(len(self.E)-1):
            self.E_avg[x]=.5*(self.E[x]+self.E[x+1])
            if (self.S[x+1]!=0):
                self.R[x]=self.S[x]/self.S[x+1]
            else:
                self.R[x]=self.S[x]/1E-20
    
    def rebin(self, bins):
        """Rebins data according to the new binning, averaging together values within the same bin

        Args:
            bins (arraylike of floats): contains the new Energy bin edges (and data will be restricted to this domain)
        
        Returns:
            (Spectra): self
        """
        SBin, _=np.histogram(self.E, bins, range=(bins[0], bins[-1]), weights=self.S)
        Ecount, _=np.histogram(self.E, bins, range=(bins[0], bins[-1]))
        SBin[np.isnan(SBin)]=0
        Emid=np.zeros(len(bins)-1)
        for i in range(len(bins)-1):
            Emid[i]=.5*(bins[i]+bins[i+1])
        self.E=Emid
        for i in range(len(Ecount)):
            if (Ecount[i]==0) or (Ecount[i]==np.nan):
                Ecount[i]=1
        self.S=SBin/Ecount
        self.binEdge=bins
        self.calcR()
        return self
    
    def integ(self, binWidth):
        """Returns the integral of the sample, with a given binwidth"""
        count=0
        S=self.S
        S[np.isnan(S)] = 0
        for x in range(len(S)):
            count+=binWidth*S[x]
        return count
    
    def renorm(self):
        """Renormalizes the spectra such that the integral underneath is 1

        Returns:
            (float): initial integral value before normalization
        """
        integ=self.integ(self.E[2]-self.E[1])
        self.S=self.S/integ
        self.calcR()
        return integ
    
    def getBinInd(self, newBins):
        """Calculates the indexes of the new binning with respect to the current energy binning
        Args:
            bins (arraylike of floats): left endpoint of each bin
        
        Returns:
            (arraylike of ints): the energy indexes for each bin
        """
        binInd=[]
        currBin=0
        i=0
        while currBin<len(newBins):
            try:
                self.binEdge[i+1]
            except:
                break
            if self.binEdge[i]<=newBins[currBin]<self.binEdge[i+1]:
                binInd.append(i)
                currBin+=1
            elif newBins[currBin]<self.binEdge[i]:
                binInd.append(i)
                currBin+=1
            i+=1
        return binInd                     
                  
    def changeFissYield(self, percOld, percNew):
        """Converts the data of 1 reactor's fission yield percentages to another's percentages using
            first order approximation using the Huber Mueller model

        Args:
            percOld (dictionary of 4 floats): fission yields associated with the old reactor
            percNew (dictionary of 4 floats): fission yields of the new reactor
                both use the keys 'u5', 'u8', 'pu9', 'pu1'

        Returns:
            Spectra: the modified object itself
        """
        hm=HuberMueller()
        binSize=self.binEdge[1]-self.binEdge[0]
        hmInd=int(10*round(binSize, 1)-1) #MIGHT NEED TO GENERALIZE THIS TO ANY BINNING
        S=np.array(self.S[:len(hm.__dict__['u5'][hmInd])])
        for i in ['u5', 'u8', 'pu9', 'pu1']:
            SpectTemp=Spectra({"E":np.array(HuberMueller.bins[hmInd][:-1]), "S": np.array(hm.__dict__[i][hmInd])})
            SpectTemp.renorm()
            
            #S+=np.array(hm.__dict__[i][int(hmInd)])*(percNew[i]-percOld[i])
            S+=SpectTemp.S*(percNew[i]-percOld[i])
        #self=redRelSpectra({"E":self.E[:len(S)], "S":list(S)})
        return sumSpectra({"E":self.E[:len(S)], "S": S})                  
 
class sumSpectra(Spectra):
    """
    Subclass of Spectra generally associated with summation objects, but also just most general
    
    Class Variables:
    fisNEOS, fisRENO, fisDAYA: dictionaries containing the fission yields of each
    crossSection: cross section value associated with energy np.arange(0, 10, .01)
    
    Attributes:
        None extra
    
    Methods:
        sumNucl: reads in a dictionary of spectra and sums it associated with a certain binning
        binDict: bins all nuclide spectra in a dictionary
        renormDict: rebins and renormalizes all nuclides in a dictionary of nuclides
        wholeSum: creates sum over a file representing nuclear data
        getFuelNuclSum: creates spectra for one of the 4 fuel nuclides (intermediary for creating whole spectra)
        reactSumSpect: calculates summation model for a specific reactor

        DEPRECATED    
        bestApprox: iterates through all options of different fission yields to find best fit to data 
    """
    #goes from 0-10, binned by .01 (the last energy term should be 9.99)
    crossSection=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 
                  0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 6.882352941178782E-5, 2.4088235294120186E-4, 4.1294117647060896E-4, 5.85000000000023E-4, 7.57058823529437E-4, 9.291176470588511E-4, 0.0011011764705882582, 0.0012732352941176722, 0.0014452941176470863, 0.0016173529411765003, 0.0017894117647059074, 0.0019614705882353214, 0.0021335294117647355, 0.0023055882352941426, 0.0024776470588235566, 0.0026497058823529707, 0.0028217647058823847, 0.002993823529411792, 0.003165882352941206, 0.003337941176470613, 0.0035100000000000166, 0.003670000000000017, 0.0038300000000000105, 0.003990000000000011, 0.004150000000000004, 0.004310000000000005, 0.004469999999999998, 0.004629999999999992, 0.004789999999999992, 0.004949999999999986, 0.005109999999999986, 0.0052699999999999796, 0.00542999999999998, 0.0055899999999999735, 0.005749999999999967, 0.005909999999999967, 0.006069999999999961, 0.006229999999999961, 0.006389999999999955, 0.006549999999999955, 0.006709999999999949, 0.006869999999999942, 0.0070299999999999425, 0.007189999999999936, 0.0073499999999999364, 0.007555769230769144, 0.007761538461538368, 0.0079673076923076, 0.008173076923076825, 0.00837884615384605, 0.008584615384615274, 0.008790384615384506, 0.00899615384615373, 0.009201923076922955, 0.009407692307692186, 0.009613461538461411, 0.009819230769230636, 0.01002499999999986, 0.010230769230769092, 0.010436538461538317, 0.010642307692307541, 0.010848076923076766, 0.011053846153845998, 0.011259615384615222, 0.011465384615384447, 0.011671153846153678, 0.011876923076922903, 0.012082692307692128, 0.012288461538461352, 0.012494230769230584, 0.012699999999999809, 0.012958620689654934, 0.013217241379310102, 0.013475862068965255, 0.013734482758620423, 0.01399310344827559, 0.014251724137930757, 0.014510344827585925, 0.014768965517241092, 0.01502758620689626, 0.015286206896551427, 0.015544827586206594, 0.01580344827586176, 0.01606206896551693, 0.016320689655172096, 0.016579310344827264, 0.01683793103448243, 0.0170965517241376, 0.017355172413792766, 0.017613793103447933, 0.0178724137931031, 0.018131034482758268, 0.018389655172413436, 0.018648275862068603, 0.01890689655172377, 0.019165517241378938, 0.019424137931034105, 0.019682758620689272, 0.01994137931034444, 0.020199999999999607, 0.020518749999999503, 0.020837499999999495, 0.021156249999999488, 0.02147499999999948, 0.021793749999999473, 0.022112499999999466, 0.02243124999999946, 0.02274999999999945, 0.023068749999999444, 0.023387499999999437, 0.02370624999999943, 0.024024999999999422, 0.024343749999999414, 0.024662499999999407, 0.0249812499999994, 0.025299999999999392, 0.025618749999999385, 0.025937499999999378, 0.02625624999999937, 0.026574999999999363, 0.026893749999999356, 0.02721249999999935, 0.02753124999999934, 0.027849999999999334, 0.028168749999999326, 0.02848749999999932, 0.02880624999999931, 0.029124999999999304, 0.029443749999999297, 0.02976249999999929, 0.030081249999999282, 0.030399999999999275, 0.030777777777776905, 0.031155555555554673, 0.03153333333333244, 0.03191111111111021, 0.03228888888888798, 0.03266666666666575, 0.033044444444443516, 0.033422222222221284, 0.03379999999999905, 0.034177777777776835, 0.034555555555554604, 0.03493333333333237, 0.03531111111111014, 0.03568888888888791, 0.03606666666666568, 0.03644444444444345, 0.036822222222221215, 0.037199999999998984, 0.03757777777777675, 0.037955555555554535, 0.03833333333333229, 0.03871111111111007, 0.03908888888888783, 0.03946666666666561, 0.039844444444443364, 0.040222222222221146, 0.04059999999999893, 0.04097777777777668, 0.041355555555554466, 0.04173333333333222, 0.04211111111111, 0.04248888888888776, 0.04286666666666554, 0.043244444444443295, 0.04362222222222108, 0.04399999999999883, 0.044436585365852335, 0.04487317073170598, 0.04530975609755962, 0.04574634146341326, 0.04618292682926693, 0.046619512195120574, 0.04705609756097422, 0.04749268292682786, 0.04792926829268153, 0.04836585365853517, 0.04880243902438881, 0.049239024390242456, 0.0496756097560961, 0.05011219512194977, 0.05054878048780341, 0.05098536585365705, 0.051421951219510695, 0.051858536585364365, 0.05229512195121801, 0.05273170731707165, 0.05316829268292529, 0.05360487804877896, 0.054041463414632604, 0.054478048780486246, 0.05491463414633989, 0.05535121951219356, 0.0557878048780472, 0.05622439024390084, 0.056660975609754485, 0.057097560975608155, 0.0575341463414618, 0.05797073170731544, 0.05840731707316908, 0.05884390243902275, 0.059280487804876394, 0.05971707317073004, 0.06015365853658368, 0.06059024390243735, 0.06102682926829099, 0.061463414634144634, 0.061899999999998276, 0.06243409090908883, 0.0629681818181797, 0.0635022727272706, 0.0640363636363615, 0.0645704545454524, 0.0651045454545433, 0.0656386363636342, 0.06617272727272511, 0.06670681818181601, 0.06724090909090691, 0.06777499999999781, 0.06830909090908871, 0.06884318181817961, 0.06937727272727051, 0.06991136363636141, 0.07044545454545231, 0.07097954545454321, 0.07151363636363411, 0.07204772727272502, 0.07258181818181589, 0.07311590909090679, 0.07364999999999769, 0.07418409090908859, 0.07471818181817949, 0.07525227272727039, 0.07578636363636129, 0.0763204545454522, 0.0768545454545431, 0.077388636363634, 0.07792272727272487, 0.07845681818181577, 0.07899090909090667, 0.07952499999999757, 0.08005909090908847, 0.08059318181817937, 0.08112727272727027, 0.08166136363636117, 0.08219545454545207, 0.08272954545454297, 0.08326363636363385, 0.08379772727272475, 0.08433181818181565, 0.08486590909090655, 0.08539999999999745, 0.08599999999999708, 0.08659999999999707, 0.08719999999999706, 0.08779999999999705, 0.08839999999999704, 0.08899999999999703, 0.08959999999999702, 0.090199999999997, 0.090799999999997, 0.09139999999999698, 0.09199999999999697, 0.09259999999999696, 0.09319999999999695, 0.09379999999999694, 0.09439999999999693, 0.09499999999999692, 0.09559999999999691, 0.0961999999999969, 0.09679999999999689, 0.09739999999999688, 0.09799999999999681, 0.0985999999999968, 0.09919999999999679, 0.09979999999999678, 0.10039999999999677, 0.10099999999999676, 0.10159999999999675, 0.10219999999999674, 0.10279999999999673, 0.10339999999999672, 0.1039999999999967, 0.1045999999999967, 0.10519999999999669, 0.10579999999999667, 0.10639999999999666, 0.10699999999999665, 0.10759999999999664, 0.10819999999999663, 0.10879999999999662, 0.10939999999999661, 0.1099999999999966, 0.11059999999999659, 0.11119999999999658, 0.11179999999999657, 0.11239999999999656, 0.11299999999999655, 0.11359999999999654, 0.11419999999999653, 0.11479999999999652, 0.1153999999999965, 0.11599999999999644, 0.11669642857142445, 0.11739285714285305, 0.1180892857142816, 0.11878571428571014, 0.11948214285713868, 0.12017857142856728, 0.12087499999999582, 0.12157142857142436, 0.1222678571428529, 0.1229642857142815, 0.12366071428571004, 0.12435714285713859, 0.12505357142856713, 0.12574999999999573, 0.12644642857142427, 0.1271428571428528, 0.12783928571428135, 0.1285357142857099, 0.1292321428571385, 0.12992857142856704, 0.13062499999999558, 0.13132142857142412, 0.13201785714285272, 0.13271428571428126, 0.1334107142857098, 0.13410714285713835, 0.13480357142856694, 0.13549999999999549, 0.13619642857142403, 0.13689285714285257, 0.13758928571428117, 0.1382857142857097, 0.13898214285713825, 0.1396785714285668, 0.1403749999999954, 0.14107142857142393, 0.14176785714285248, 0.14246428571428102, 0.14316071428570962, 0.14385714285713816, 0.1445535714285667, 0.14524999999999524, 0.14594642857142384, 0.14664285714285238, 0.14733928571428093, 0.14803571428570947, 0.14873214285713807, 0.1494285714285666, 0.15012499999999515, 0.1508214285714237, 0.15151785714285224, 0.15221428571428083, 0.15291071428570938, 0.15360714285713792, 0.15430357142856646, 0.15499999999999506, 0.15580645161289747, 0.15661290322580068, 0.1574193548387039, 0.1582258064516071, 0.15903225806451032, 0.15983870967741354, 0.16064516129031675, 0.1614516129032199, 0.16225806451612312, 0.16306451612902634, 0.16387096774192955, 0.16467741935483277, 0.16548387096773598, 0.1662903225806392, 0.1670967741935424, 0.16790322580644562, 0.16870967741934884, 0.16951612903225205, 0.17032258064515526, 0.17112903225805842, 0.17193548387096164, 0.17274193548386485, 0.17354838709676806, 0.17435483870967128, 0.1751612903225745, 0.1759677419354777, 0.17677419354838092, 0.17758064516128413, 0.17838709677418735, 0.17919354838709056, 0.17999999999999378, 0.180806451612897, 0.18161290322580015, 0.18241935483870336, 0.18322580645160658, 0.1840322580645098, 0.184838709677413, 0.18564516129031622, 0.18645161290321943, 0.18725806451612265, 0.18806451612902586, 0.18887096774192907, 0.1896774193548323, 0.1904838709677355, 0.19129032258063866, 0.19209677419354187, 0.1929032258064451, 0.1937096774193483, 0.19451612903225152, 0.19532258064515473, 0.19612903225805794, 0.19693548387096116, 0.19774193548386437, 0.1985483870967676, 0.1993548387096708, 0.20016129032257401, 0.20096774193547717, 0.2017741935483804, 0.2025806451612836, 0.20338709677418682, 0.20419354838709003, 0.20499999999999324, 0.20591428571427806, 0.2068285714285638, 0.20774285714284946, 0.2086571428571351, 0.20957142857142086, 0.2104857142857065, 0.21139999999999226, 0.2123142857142779, 0.21322857142856366, 0.2141428571428493, 0.21505714285713506, 0.2159714285714207, 0.21688571428570635, 0.2177999999999921, 0.21871428571427776, 0.2196285714285635, 0.22054285714284916, 0.22145714285713491, 0.22237142857142056, 0.22328571428570632, 0.22419999999999196, 0.2251142857142776, 0.22602857142856336, 0.226942857142849, 0.22785714285713476, 0.2287714285714204, 0.22968571428570617, 0.2305999999999918, 0.23151428571427757, 0.2324285714285632, 0.23334285714284886, 0.23425714285713461, 0.23517142857142026, 0.23608571428570602, 0.23699999999999166, 0.23791428571427742, 0.23882857142856306, 0.23974285714284882, 0.24065714285713447, 0.24157142857142022, 0.24248571428570587, 0.2433999999999915, 0.24431428571427727, 0.2452285714285629, 0.24614285714284867, 0.24705714285713432, 0.24797142857142007, 0.24888571428570572, 0.24979999999999147, 0.2507142857142771, 0.25162857142856276, 0.2525428571428485, 0.25345714285713417, 0.2543714285714199, 0.25528571428570557, 0.2561999999999913, 0.25711428571427697, 0.2580285714285627, 0.25894285714284837, 0.259857142857134, 0.26077142857141977, 0.2616857142857054, 0.2625999999999912, 0.2635142857142768, 0.2644285714285626, 0.2653428571428482, 0.266257142857134, 0.2671714285714196, 0.26808571428570527, 0.268999999999991, 0.27003896103895075, 0.2710779220779118, 0.2721168831168729, 0.2731558441558338, 0.2741948051947949, 0.27523376623375584, 0.2762727272727169, 0.27731168831167796, 0.2783506493506389, 0.2793896103896, 0.2804285714285609, 0.281467532467522, 0.28250649350648294, 0.283545454545444, 0.28458441558440506, 0.285623376623366, 0.28666233766232707, 0.287701298701288, 0.2887402597402491, 0.28977922077921014, 0.2908181818181711, 0.29185714285713216, 0.2928961038960931, 0.29393506493505417, 0.29497402597401523, 0.2960129870129762, 0.29705194805193724, 0.2980909090908982, 0.29912987012985925, 0.3001688311688202, 0.30120779220778127, 0.30224675324674233, 0.3032857142857033, 0.30432467532466434, 0.3053636363636253, 0.30640259740258635, 0.3074415584415474, 0.30848051948050836, 0.3095194805194694, 0.3105584415584304, 0.31159740259739144, 0.3126363636363525, 0.31367532467531345, 0.3147142857142745, 0.31575324675323546, 0.3167922077921965, 0.3178311688311575, 0.31887012987011853, 0.3199090909090796, 0.32094805194804055, 0.3219870129870016, 0.32302597402596256, 0.3240649350649236, 0.3251038961038847, 0.32614285714284563, 0.3271818181818067, 0.32822077922076764, 0.3292597402597287, 0.33029870129868966, 0.3313376623376507, 0.3323766233766118, 0.33341558441557273, 0.3344545454545338, 0.33549350649349474, 0.3365324675324558, 0.33757142857141686, 0.3386103896103778, 0.3396493506493389, 0.3406883116882998, 0.3417272727272609, 0.34276623376622195, 0.3438051948051829, 0.34484415584414396, 0.3458831168831049, 0.346922077922066, 0.3479610389610269, 0.348999999999988, 0.35017241379308983, 0.35134482758619334, 0.35251724137929674, 0.35368965517240014, 0.35486206896550354, 0.35603448275860705, 0.35720689655171045, 0.35837931034481385, 0.35955172413791725, 0.36072413793102065, 0.36189655172412416, 0.36306896551722756, 0.36424137931033096, 0.36541379310343436, 0.36658620689653787, 0.36775862068964127, 0.36893103448274467, 0.37010344827584807, 0.3712758620689516, 0.372448275862055, 0.3736206896551584, 0.3747931034482618, 0.3759655172413652, 0.3771379310344687, 0.3783103448275721, 0.3794827586206755, 0.3806551724137789, 0.3818275862068824, 0.3829999999999858, 0.3841724137930892, 0.3853448275861926, 0.386517241379296, 0.3876896551723995, 0.3888620689655029, 0.3900344827586063, 0.3912068965517097, 0.3923793103448132, 0.3935517241379166, 0.39472413793102, 0.3958965517241234, 0.3970689655172269, 0.3982413793103303, 0.3994137931034337, 0.4005862068965371, 0.4017586206896405, 0.40293103448274403, 0.40410344827584743, 0.40527586206895083, 0.40644827586205423, 0.40762068965515774, 0.40879310344826114, 0.40996551724136454, 0.41113793103446794, 0.41231034482757134, 0.41348275862067485, 0.41465517241377825, 0.41582758620688165, 0.41699999999998505, 0.41817241379308856, 0.41934482758619196, 0.42051724137929536, 0.42168965517239876, 0.42286206896550227, 0.42403448275860567, 0.42520689655170907, 0.42637931034481247, 0.42755172413791587, 0.4287241379310194, 0.4298965517241228, 0.4310689655172262, 0.4322413793103296, 0.4334137931034331, 0.4345862068965365, 0.4357586206896399, 0.4369310344827433, 0.4381034482758467, 0.4392758620689502, 0.4404482758620536, 0.441620689655157, 0.4427931034482604, 0.4439655172413639, 0.4451379310344673, 0.4463103448275707, 0.4474827586206741, 0.4486551724137776, 0.449827586206881, 0.4509999999999844, 0.45227659574466383, 0.4535531914893446, 0.45482978723402556, 0.4561063829787063, 0.45738297872338707, 0.45865957446806804, 0.4599361702127488, 0.46121276595742955, 0.4624893617021103, 0.4637659574467913, 0.46504255319147203, 0.4663191489361528, 0.46759574468083376, 0.4688723404255145, 0.47014893617019526, 0.47142553191487624, 0.472702127659557, 0.47397872340423775, 0.4752553191489187, 0.4765319148935995, 0.4778085106382802, 0.4790851063829612, 0.48036170212764195, 0.4816382978723227, 0.48291489361700346, 0.48419148936168444, 0.4854680851063652, 0.48674468085104594, 0.4880212765957269, 0.48929787234040767, 0.4905744680850884, 0.4918510638297694, 0.49312765957445015, 0.4944042553191309, 0.4956808510638119, 0.49695744680849263, 0.4982340425531734, 0.49951063829785436, 0.5007872340425351, 0.5020638297872159, 0.5033404255318966, 0.5046170212765776, 0.5058936170212583, 0.5071702127659391, 0.5084468085106201, 0.5097234042553008, 0.5109999999999816, 0.5124019607842935, 0.5138039215686071, 0.515205882352921, 0.5166078431372346, 0.5180098039215483, 0.5194117647058619, 0.5208137254901756, 0.5222156862744894, 0.5236176470588031, 0.5250196078431167, 0.5264215686274304, 0.527823529411744, 0.5292254901960579, 0.5306274509803716, 0.5320294117646852, 0.5334313725489989, 0.5348333333333127, 0.5362352941176264, 0.53763725490194, 0.5390392156862537, 0.5404411764705673, 0.5418431372548812, 0.5432450980391949, 0.5446470588235085, 0.5460490196078222, 0.5474509803921358, 0.5488529411764497, 0.5502549019607633, 0.551656862745077, 0.5530588235293906, 0.5544607843137043, 0.5558627450980181, 0.5572647058823318, 0.5586666666666454, 0.5600686274509591, 0.561470588235273, 0.5628725490195866, 0.5642745098039003, 0.5656764705882139, 0.5670784313725276, 0.5684803921568414, 0.5698823529411551, 0.5712843137254687, 0.5726862745097824, 0.574088235294096, 0.5754901960784099, 0.5768921568627235, 0.5782941176470372, 0.5796960784313508, 0.5810980392156647, 0.5824999999999784, 0.583901960784292, 0.5853039215686057, 0.5867058823529193, 0.5881078431372332, 0.5895098039215468, 0.5909117647058605, 0.5923137254901741, 0.5937156862744878, 0.5951176470588017, 0.5965196078431153, 0.597921568627429, 0.5993235294117426, 0.6007254901960563, 0.6021274509803701, 0.6035294117646838, 0.6049313725489974, 0.6063333333333111, 0.607735294117625, 0.6091372549019386, 0.6105392156862522, 0.6119411764705659, 0.6133431372548795, 0.6147450980391934, 0.6161470588235071, 0.6175490196078207, 0.6189509803921344, 0.620352941176448, 0.6217549019607619, 0.6231568627450755, 0.6245588235293892, 0.6259607843137028, 0.6273627450980167, 0.6287647058823304, 0.630166666666644, 0.6315686274509577, 0.6329705882352713, 0.6343725490195852, 0.6357745098038988, 0.6371764705882125, 0.6385784313725261, 0.6399803921568398, 0.6413823529411536, 0.6427843137254673, 0.6441862745097809, 0.6455882352940946, 0.6469901960784082, 0.6483921568627221, 0.6497941176470358, 0.6511960784313494, 0.6525980392156631, 0.6539999999999769, 0.6555478260869309, 0.6570956521738874, 0.658643478260844, 0.6601913043478004, 0.661739130434757, 0.6632869565217133, 0.6648347826086699, 0.6663826086956263, 0.6679304347825828, 0.6694782608695394, 0.6710260869564958, 0.6725739130434524, 0.6741217391304087, 0.6756695652173653, 0.6772173913043217, 0.6787652173912783, 0.6803130434782348, 0.6818608695651912, 0.6834086956521478, 0.6849565217391042, 0.6865043478260607, 0.6880521739130171, 0.6895999999999737, 0.6911478260869303, 0.6926956521738866, 0.6942434782608432, 0.6957913043477996, 0.6973391304347561, 0.6988869565217125, 0.7004347826086691, 0.7019826086956257, 0.703530434782582, 0.7050782608695386, 0.706626086956495, 0.7081739130434516, 0.7097217391304079, 0.7112695652173645, 0.7128173913043211, 0.7143652173912775, 0.715913043478234, 0.7174608695651904, 0.719008695652147, 0.7205565217391033, 0.7221043478260599, 0.7236521739130165, 0.7251999999999729, 0.7267478260869294, 0.7282956521738858, 0.7298434782608424, 0.7313913043477988, 0.7329391304347553, 0.7344869565217119, 0.7360347826086683, 0.7375826086956249, 0.7391304347825812, 0.7406782608695378, 0.7422260869564942, 0.7437739130434508, 0.7453217391304073, 0.7468695652173637, 0.7484173913043203, 0.7499652173912766, 0.7515130434782332, 0.7530608695651896, 0.7546086956521462, 0.7561565217391027, 0.7577043478260591, 0.7592521739130157, 0.760799999999972, 0.7623478260869286, 0.763895652173885, 0.7654434782608416, 0.7669913043477982, 0.7685391304347545, 0.7700869565217111, 0.7716347826086675, 0.773182608695624, 0.7747304347825804, 0.776278260869537, 0.7778260869564936, 0.7793739130434499, 0.7809217391304065, 0.7824695652173629, 0.7840173913043195, 0.7855652173912758, 0.7871130434782324, 0.788660869565189, 0.7902086956521454, 0.7917565217391019, 0.7933043478260583, 0.7948521739130149, 0.7963999999999712, 0.7979478260869278, 0.7994956521738844, 0.8010434782608408, 0.8025913043477974, 0.8041391304347537, 0.8056869565217103, 0.8072347826086667, 0.8087826086956232, 0.8103304347825798, 0.8118782608695362, 0.8134260869564928, 0.8149739130434491, 0.8165217391304057, 0.8180695652173621, 0.8196173913043187, 0.8211652173912752, 0.8227130434782316, 0.8242608695651882, 0.8258086956521445, 0.8273565217391011, 0.8289043478260575, 0.8304521739130141, 0.8319999999999707, 0.833676923076891, 0.8353538461538141, 0.837030769230737, 0.8387076923076602, 0.8403846153845833, 0.8420615384615062, 0.8437384615384294, 0.8454153846153523, 0.8470923076922754, 0.8487692307691985, 0.8504461538461214, 0.8521230769230446, 0.8537999999999675, 0.8554769230768906, 0.8571538461538135, 0.8588307692307366, 0.8605076923076598, 0.8621846153845827, 0.8638615384615058, 0.8655384615384287, 0.8672153846153519, 0.868892307692275, 0.8705692307691979, 0.872246153846121, 0.8739230769230439, 0.8755999999999671, 0.8772769230768902, 0.8789538461538131, 0.8806307692307362, 0.8823076923076592, 0.8839846153845823, 0.8856615384615054, 0.8873384615384283, 0.8890153846153515, 0.8906923076922744, 0.8923692307691975, 0.8940461538461206, 0.8957230769230435, 0.8973999999999667, 0.8990769230768896, 0.9007538461538127, 0.9024307692307356, 0.9041076923076588, 0.9057846153845819, 0.9074615384615048, 0.9091384615384279, 0.9108153846153508, 0.912492307692274, 0.9141692307691971, 0.91584615384612, 0.9175230769230431, 0.919199999999966, 0.9208769230768892, 0.9225538461538123, 0.9242307692307352, 0.9259076923076583, 0.9275846153845813, 0.9292615384615044, 0.9309384615384275, 0.9326153846153504, 0.9342923076922736, 0.9359692307691965, 0.9376461538461196, 0.9393230769230427, 0.9409999999999656, 0.9426769230768888, 0.9443538461538117, 0.9460307692307348, 0.9477076923076577, 0.9493846153845809, 0.951061538461504, 0.9527384615384269, 0.95441538461535, 0.9560923076922729, 0.9577692307691961, 0.9594461538461192, 0.9611230769230421, 0.9627999999999652, 0.9644769230768881, 0.9661538461538113, 0.9678307692307344, 0.9695076923076573, 0.9711846153845805, 0.9728615384615034, 0.9745384615384265, 0.9762153846153496, 0.9778923076922725, 0.9795692307691957, 0.9812461538461186, 0.9829230769230417, 0.9845999999999648, 0.9862769230768877, 0.9879538461538109, 0.989630769230734, 0.9913076923076567, 0.9929846153845798, 0.994661538461503, 0.9963384615384261, 0.9980153846153492, 0.9996923076922719, 1.001369230769195, 1.0030461538461182, 1.0047230769230413, 1.0063999999999644, 1.0080769230768871, 1.0097538461538103, 1.0114307692307334, 1.0131076923076565, 1.0147846153845796, 1.0164615384615023, 1.0181384615384255, 1.0198153846153486, 1.0214923076922717, 1.0231692307691949, 1.0248461538461175, 1.0265230769230407, 1.0281999999999638, 1.029876923076887, 1.03155384615381, 1.0332307692307328, 1.034907692307656, 1.036584615384579, 1.0382615384615022, 1.0399384615384253, 1.041615384615348, 1.043292307692271, 1.0449692307691942, 1.0466461538461174, 1.0483230769230405, 1.0499999999999632, 1.0519999999999565, 1.0539999999999563, 1.0559999999999565, 1.0579999999999563, 1.059999999999956, 1.0619999999999563, 1.063999999999956, 1.0659999999999563, 1.067999999999956, 1.0699999999999563, 1.071999999999956, 1.0739999999999559, 1.075999999999956, 1.0779999999999559, 1.079999999999956, 1.0819999999999559, 1.0839999999999557, 1.085999999999956, 1.0879999999999557, 1.089999999999956, 1.0919999999999557, 1.0939999999999555, 1.0959999999999557, 1.0979999999999555, 1.0999999999999557, 1.1019999999999555, 1.1039999999999552, 1.1059999999999555, 1.1079999999999552, 1.1099999999999555, 1.1119999999999552, 1.113999999999955, 1.1159999999999553, 1.117999999999955, 1.1199999999999553, 1.121999999999955, 1.1239999999999548, 1.125999999999955, 1.1279999999999548, 1.129999999999955, 1.1319999999999548, 1.1339999999999546, 1.1359999999999548, 1.1379999999999546, 1.1399999999999548, 1.1419999999999546, 1.1439999999999548, 1.1459999999999546, 1.1479999999999544, 1.1499999999999546, 1.1519999999999544, 1.1539999999999546, 1.1559999999999544, 1.1579999999999542, 1.1599999999999544, 1.1619999999999542, 1.1639999999999544, 1.1659999999999542, 1.167999999999954, 1.1699999999999542, 1.171999999999954, 1.1739999999999542, 1.175999999999954, 1.1779999999999538, 1.179999999999954, 1.1819999999999538, 1.183999999999954, 1.1859999999999538, 1.1879999999999535])
    
    fisNEOS={"u5":0.655, "u8":0.072, "pu9":0.235, "pu1":0.038}
    fisRENO={"u5":0.571, "u8":0.073, "pu9":0.300, "pu1":0.056}
    fisDAYA={"u5": 0.564, "u8": .076, "pu9":.304, "pu1": 0.056}
    
    def __init__(self, dict, ifSimp=False):
        super().__init__(dict, ifSimp=ifSimp)
    
    @staticmethod
    def sumNucl(dict, bins=None, weight=None):
        """
        Sums spectra over nuclide dictionary, binning data in the process

        Args:
            dict (dictionary of Spectra): dictionary of spectra to be summed
            bins (array, optional): bin edges for the data to be placed into. Defaults to Util.getBin default
            weight (dictionary of floats): contains key equal to a given nuclide code, and value of its associated weight
                Defaults to None, implying that the elements are already preweighted

        Returns:
            sumSpectra: sumspectra with E=Emid, S=sum
        """
        bins=Util.getBin(bins)
        Emid=np.zeros(len(bins)-1)
        if isinstance(weight, type(None)):
            weight={}
            for x in dict.keys():
                weight[x]=1
        dict=sumSpectra.binDict(dict, bins)
        Ecount=np.zeros(len(bins)-1)
        for x in range(len(bins)-1):
            Emid[x]=.5*(bins[x]+bins[x+1])
        sumS=np.zeros(len(bins)-1)
        count=0
        for nucl in dict:
            count+=1
            SBin, _=np.histogram(dict[nucl].E, bins, range=(bins[0], bins[-1]), weights=dict[nucl].S*weight[nucl])
            Ecount, _=np.histogram(dict[nucl].E, bins, range=(bins[0], bins[-1]))
            SBin[np.isnan(SBin)]=0
            for x in range(len(Ecount)):
                if Ecount[x]==0:
                    Ecount[x]=1
            sumS+=SBin/Ecount
        return sumSpectra({"E":Emid, "S":sumS})
    
    @staticmethod
    def binDict(dict, bins):
        """Bins all the Nuclide spectra in the dictionary to the bins specified"""
        for key in dict:
            dict[key]=dict[key].rebin(bins)
        return dict
    
    @staticmethod
    def renormDict(dict, bins=None):
        """
        Renormalizes the dictionary of nuclideSpectra such that their summation integral is 1

        Args:
            dict (dictionary of nuclideSpectra): dictionary of the nuclideSpectra
            bins (arraylike of floats, optional): binning of the data. Defaults to Util.getBin default

        Returns:
            (dictionary of nuclideSpectra): renormalized dictionary
            (float): integral over all the sources before renormalization
        """
        bins=Util.getBin(bins)
        sumS=sumSpectra.sumNucl(dict, bins=bins)
        integ=sumS.renorm()
        inte={}
        for x in dict:
            tempD=dict[x].__dict__
            tempD["S"]/=integ
            dict[x]=sumSpectra(tempD)
            inte[x]=dict[x].integ(bins[2]-bins[1])
        return dict, inte
    
    @staticmethod
    def wholeSum(filePath, bins=None):
        """Creates sumSpectra element with the entire dictionary of spectra"""        
        bins=Util.getBin(bins)
        return sumSpectra.sumNucl(Util.readDict(nuclideSpectra, filePath, saveFile="Nuclide Spectral Data\\high res NEOS nuclides\\"), bins=bins)
        
    @staticmethod
    def getFuelNuclSum(nucl, bins=None):
        """
        Determines the summation spectra for one of the 4 fuel nuclides; 235U, 238U, 239Pu, 241Pu 
            by adding up the contributions from each nuclide

        Args:
            nucl (string): in one of the following forms: u5, u8, pu9, pu1
            bins (arraylike of floats, optional): the bin edges used. Defaults to np.arange(0, 10.01, .01).

        Returns:
            Sum Spectra: CFY for the specific nucl
        """
        fileName="Nuclide Spectral Data\\" +nucl+"cfy.txt"
        with open(fileName, 'r') as f:
            data=f.read()
        strArr=Util.strToStrArr(data, delimeter=";", numCols=4)
        dictWeight={}
        for i in range(len(strArr[0])):
            strKey="z_"+str(strArr[0][i])+"_a_"+str(strArr[1][i])+"_liso_"+str(strArr[2][i])
            if strKey not in nuclideSpectra.names:
                strKey+="_t"
            dictWeight[strKey]=float(strArr[3][i])
        if isinstance(bins, type(None)):
            bins=np.arange(0, 10.01, .01)
        return sumSpectra.sumNucl(Util.readDict(nuclideSpectra, filePath=r"rawNuclideSpectra.csv"), bins, weight=dictWeight)
    
    @staticmethod
    def reactSumSpect(percent, ifRaw=False):
        """
        Creates a sum spectra for a reactor based on the specific percentages of u5, u8, pu9, and pu1

        Args:
            percent (dictionary of 4 floats): percentage  for each nuclide with key being name as above
            ifRaw (boolean, optional): if true, conducts the entire calculation from scratch by calculating 
                each of the individual contributions. Defaults to False (draws from a file location)
            
        Returns:
            Sum Spectra: total summation for the reactor
        """
        if ifRaw:
            count=0
            E, S=None, None
            for x in ["u5", "u8", "pu9", "pu1"]:
                nuclSpect=sumSpectra.getFuelNuclSum(x)
                if count==0:
                    E=nuclSpect.E
                    S=nuclSpect.S*percent[x]
                    count+=1
                else:
                    S+=nuclSpect.S*percent[x]
            ranE=(E[0], E[-1])
            inds=Util.getIndex(ranE, bins=np.arange(0, 10, .01))
            S*=np.array(sumSpectra.crossSection[:1000])
        else:
            dict=Util.readDict(sumSpectra, "fuelSpectra.csv")
            u5, u8, pu9, pu1=dict["u5"], dict["u8"], dict["pu9"], dict["pu1"]
            E=u5.E
            S=percent["u5"]*np.array(u5.S)+percent["u8"]*np.array(u8.S)+percent["pu9"]*np.array(pu9.S)+percent["pu1"]*np.array(pu1.S)
        return sumSpectra({"E": E, "S": S})
                
    @staticmethod
    def bestApprox(data, bins=None, sampWidth=.1, meas="S", plotHM=False, split=.5, fuel2=["u5", "pu9"], ifAnim=False, lims=[1.8, 8], ifLS=False):
        """
        Outputs the closest fit to the data from summations of the 4 spectra
            Tests all the different permutations to find the closest fit
        
        Args:
            data (Spectra): contains the spectra to be fitted to
            bins (arraylike of floats, optional): contains the bin spacing to be used for the data analysis. Defaults to Util.getBin default 
            sampWidth (float, optional): the width of the spacing used for the search. Defaults to .1
            meas (string, optional): the comparison method being used; CURRENTLY ONLY S available
            plotHM (boolean, optional): if True, makes a heat map with regards to 2 of the fuel sources
                setting the other two to split the remaining amount evenly. Defaults to False
            split (float, optional): the proportional splitting in remaining amount between the two excluded samples. 
                Defaults to .5
            fuel2 (array of 2 strings, optional): the two fuel nuclides to span the whole spectrum. Defaults to u5 and pu9
            ifAnim (boolean, optional): if true, makes an animation by sending the individual graphs back to an updating method. Defaults to False
            lims (arraylike of 2 floats, optional): the energy bounds for the chi-squared calculation. Defaults to [1.8, 8]
            ifLS (boolean, optional): if true, uses Least Squares instead of chi-squared for the distance between plots.
            
        Returns:
            (dictionary of 4 floats): relative proportions of u5, u8, pu9, pu1 that minimizes the chi^2 value
            (float): Chi square value that minimizes
        """          
        dict=Util.readDict(sumSpectra, "fuelSpectra.csv")
        u5, u8, pu9, pu1=dict["u5"], dict["u8"], dict["pu9"], dict["pu1"]
        nameArr=["u5", "u8", "pu9", "pu1"]
        arrInd=[]
        for x in fuel2:
            arrInd.append(nameArr.index(x))
        arrOut=[]
        for x in range(4):
            if x not in arrInd:
                arrOut.append(x)
        u5.rebin(bins)
        u5.renorm()
        u8.rebin(bins)
        u8.renorm()
        pu9.rebin(bins)
        pu9.renorm()
        pu1.rebin(bins)
        pu1.renorm()
        match meas:
            case "S":
                arr5=u5.S
                arr8=u8.S
                arr1=pu1.S
                arr9=pu9.S
                data.rebin(bins)
                data.renorm()
                comp=data.S

        chiSq=9999999999999 #arbitrarily large starting value
        finVals=[]
        count=0
        if plotHM:
            
            #fig, axs=plt.subplots(2, 3)
            arr=np.zeros((int(1/sampWidth), int(1/sampWidth)))
            counti=0
            countj=0
            for i in np.arange(stop=1, step=sampWidth): #u5
                countj=0
                for j in np.arange(stop=1-i, step=sampWidth): #u8
                            vals=[0, 0, 0, 0]
                            vals[arrInd[0]]=i
                            vals[arrInd[1]]=j
                            vals[arrOut[0]]=split*(1-i-j)
                            vals[arrOut[1]]=(1-split)*(1-i-j)
                    #for k in np.arange(stop=1-i-j, step=sampWidth): #pu9
                        #for l in np.arange(stop=1-i-j-k, step=sampWidth): #pu1
                            sq=0
                            if count%1000==0:
                                print("Best Approx Count: "+count)
                            count+=1
                            match meas:
                                case "S":
                                    sq, _=Util.chiSq(comp,vals[0]*arr5+vals[1]*arr8+
                                                vals[2]*arr9+vals[3]*arr1, ifLS=ifLS, lims=lims)
                            arr[counti, countj]=sq
                            if sq<chiSq:
                                chiSq=sq
                                finVals=vals
                            countj+=1
                counti+=1
            if ifAnim:
                return arr
            c=plt.imshow(arr, cmap="viridis", origin="lower", interpolation="nearest", norm=matplotlib.colors.Normalize(.001, .02, clip=True))
            plt.colorbar()
            plt.xlabel("Rel amount of "+fuel2[0])
            plt.ylabel("Rel amount of u238 "+fuel2[1])
            excl=[nameArr[i] for i in arrOut]
            st=""
            for x in excl:
                st+=x+" "
            plt.title("Excluded elements with split "+str(split) + " : " +st)
            plt.show()
        else:
            for i in np.arange(stop=1, step=sampWidth): #u5
                for j in np.arange(stop=1-i, step=sampWidth): #u8
                    for k in np.arange(stop=1-i-j, step=sampWidth): #pu9
                        for l in np.arange(stop=1-i-j-k, step=sampWidth): #pu1
                            sq=0
                            if count%1000==0:
                                print("Best Approx count: "+count)
                            count+=1
                            match meas:
                                case "S":
                                    sq, _=Util.chiSq(comp,i*arr5+j*arr8+
                                                k*arr9+l*arr1, ifLS=ifLS, lims=lims)
                            if sq<chiSq:
                                chiSq=sq
                                finVals=[i, j, k, l]
        return {"u5":finVals[0], "u8": finVals[1], "pu9": finVals[2], "pu1": finVals[3]}, chiSq
                               
class nuclideSpectra(Spectra):
    """Subclass of spectra associated with specific nuclide, often in dictionary form

    Class Variables:
        names (array of strings): the names of all the different nuclides of interest, in standard nomenclature

    Attributes:
        Z (int): atomic number
        A (int): atomic mass
        liso (int): level of isomer
        epE (float): endpoint energy [MeV]
        ifT (boolean): if theoretical
        name (string): follows format "z_#_a_##_liso_###(_t)"
            # is Z, ## is A, ### is liso, and (_t) included if ifT==True
        disc (array of floats): discontinuities in the spectra
        
    Class Methods:
        getVals: reads info from a nuclear spectra file
        makeDatabase: creates database of nuclear spectra for easy conversion to dictionary of nuclideSpectra object
        makeReactDatbase: same as makeDatabase but scales all spectra according to reactor CFY
        getName: outputs name in official nomenclature (if not set on initialization)
    """
    names=['z_1_a_3_liso_0', 'z_24_a_56_liso_0', 'z_24_a_57_liso_0', 'z_25_a_56_liso_0', 
           'z_25_a_57_liso_0', 'z_25_a_58_liso_0', 'z_25_a_59_liso_0', 'z_25_a_60_liso_0', 'z_25_a_61_liso_0', 'z_25_a_62_liso_0', 'z_26_a_59_liso_0', 'z_26_a_60_liso_0', 'z_26_a_61_liso_0', 'z_26_a_62_liso_0', 'z_26_a_63_liso_0', 'z_26_a_64_liso_0', 'z_26_a_65_liso_0', 'z_27_a_61_liso_0', 'z_27_a_62_liso_0', 'z_27_a_63_liso_0', 'z_27_a_64_liso_0', 'z_27_a_65_liso_0', 'z_27_a_66_liso_0', 'z_27_a_67_liso_0', 'z_27_a_68_liso_0', 'z_27_a_72_liso_0_t', 'z_27_a_73_liso_0_t', 'z_27_a_74_liso_0_t', 'z_28_a_63_liso_0', 'z_28_a_65_liso_0', 'z_28_a_66_liso_0', 'z_28_a_67_liso_0', 'z_28_a_69_liso_0', 'z_28_a_72_liso_0_t', 'z_28_a_73_liso_0_t', 'z_28_a_74_liso_0_t', 'z_28_a_75_liso_0_t', 'z_28_a_76_liso_0_t', 'z_28_a_77_liso_0_t', 'z_29_a_66_liso_0', 'z_29_a_67_liso_0', 'z_29_a_68_liso_0', 'z_29_a_69_liso_0', 'z_29_a_70_liso_0', 'z_29_a_72_liso_0', 'z_29_a_73_liso_0', 'z_29_a_74_liso_0', 'z_29_a_75_liso_0_t', 'z_29_a_76_liso_0', 'z_29_a_77_liso_0_t', 'z_29_a_78_liso_0_t', 'z_29_a_79_liso_0_t', 'z_29_a_80_liso_0_t', 'z_2_a_6_liso_0', 'z_30_a_69_liso_0', 'z_30_a_69_liso_1', 'z_30_a_71_liso_0', 'z_30_a_71_liso_1', 'z_30_a_72_liso_0', 'z_30_a_73_liso_0', 'z_30_a_74_liso_0_t', 'z_30_a_75_liso_0', 'z_30_a_76_liso_0', 'z_30_a_77_liso_0', 'z_30_a_78_liso_0', 'z_30_a_79_liso_0_t', 'z_30_a_80_liso_0_t', 'z_30_a_81_liso_0_t', 'z_30_a_82_liso_0_t', 'z_31_a_72_liso_0', 'z_31_a_73_liso_0', 'z_31_a_74_liso_0', 'z_31_a_74_liso_1', 'z_31_a_75_liso_0', 'z_31_a_76_liso_0', 'z_31_a_77_liso_0_t', 'z_31_a_78_liso_0', 'z_31_a_79_liso_0', 'z_31_a_80_liso_0', 'z_31_a_81_liso_0', 'z_31_a_82_liso_0_t', 'z_31_a_83_liso_0_t', 'z_31_a_84_liso_0_t', 'z_31_a_85_liso_0_t', 'z_32_a_75_liso_0', 'z_32_a_75_liso_1', 'z_32_a_77_liso_0', 'z_32_a_77_liso_1', 'z_32_a_78_liso_0', 'z_32_a_79_liso_0', 'z_32_a_79_liso_1', 'z_32_a_80_liso_0', 'z_32_a_81_liso_0', 'z_32_a_81_liso_1', 'z_32_a_82_liso_0', 'z_32_a_83_liso_0', 'z_32_a_84_liso_0_t', 'z_32_a_85_liso_0_t', 'z_32_a_86_liso_0_t', 'z_32_a_87_liso_0_t', 'z_33_a_76_liso_0', 'z_33_a_77_liso_0', 'z_33_a_78_liso_0', 'z_33_a_79_liso_0', 'z_33_a_80_liso_0', 'z_33_a_81_liso_0', 'z_33_a_82_liso_0', 'z_33_a_82_liso_1', 'z_33_a_83_liso_0_t', 'z_33_a_84_liso_0', 'z_33_a_85_liso_0_t', 'z_33_a_86_liso_0_t', 'z_33_a_87_liso_0_t', 'z_33_a_88_liso_0_t', 'z_33_a_89_liso_0_t', 'z_33_a_90_liso_0_t', 'z_34_a_79_liso_0', 'z_34_a_81_liso_0', 'z_34_a_81_liso_1', 'z_34_a_83_liso_0', 'z_34_a_83_liso_1', 'z_34_a_84_liso_0', 'z_34_a_85_liso_0', 'z_34_a_86_liso_0', 'z_34_a_87_liso_0', 'z_34_a_88_liso_0', 'z_34_a_89_liso_0_t', 'z_34_a_90_liso_0_t', 'z_34_a_91_liso_0_t', 'z_34_a_92_liso_0_t', 'z_35_a_80_liso_0', 'z_35_a_82_liso_0', 'z_35_a_82_liso_1', 'z_35_a_83_liso_0', 'z_35_a_84_liso_0', 'z_35_a_84_liso_1', 'z_35_a_85_liso_0', 'z_35_a_86_liso_0', 'z_35_a_87_liso_0', 'z_35_a_88_liso_0', 'z_35_a_89_liso_0', 'z_35_a_90_liso_0', 'z_35_a_91_liso_0', 'z_35_a_92_liso_0_t', 'z_35_a_93_liso_0_t', 'z_35_a_94_liso_0_t', 'z_35_a_95_liso_0_t', 'z_36_a_85_liso_0', 'z_36_a_85_liso_1', 'z_36_a_87_liso_0', 'z_36_a_88_liso_0', 'z_36_a_89_liso_0', 'z_36_a_90_liso_0', 'z_36_a_91_liso_0', 'z_36_a_92_liso_0', 'z_36_a_93_liso_0', 'z_36_a_94_liso_0_t', 'z_36_a_95_liso_0_t', 'z_36_a_96_liso_0_t', 'z_36_a_97_liso_0_t', 'z_36_a_98_liso_0_t', 'z_37_a_100_liso_0_t', 'z_37_a_86_liso_0', 'z_37_a_87_liso_0', 'z_37_a_88_liso_0', 'z_37_a_89_liso_0', 'z_37_a_90_liso_0', 'z_37_a_90_liso_1', 'z_37_a_91_liso_0', 'z_37_a_92_liso_0', 'z_37_a_93_liso_0', 'z_37_a_94_liso_0', 'z_37_a_95_liso_0', 'z_37_a_96_liso_0', 'z_37_a_97_liso_0', 'z_37_a_98_liso_0', 'z_37_a_99_liso_0_t', 'z_38_a_100_liso_0', 'z_38_a_101_liso_0_t', 'z_38_a_102_liso_0_t', 'z_38_a_103_liso_0_t', 'z_38_a_89_liso_0', 'z_38_a_90_liso_0', 'z_38_a_91_liso_0', 'z_38_a_92_liso_0', 'z_38_a_93_liso_0', 'z_38_a_94_liso_0', 'z_38_a_95_liso_0', 'z_38_a_96_liso_0', 'z_38_a_97_liso_0', 'z_38_a_98_liso_0', 'z_38_a_99_liso_0', 'z_39_a_100_liso_0', 'z_39_a_101_liso_0_t', 'z_39_a_102_liso_0_t', 'z_39_a_102_liso_1_t', 'z_39_a_103_liso_0_t', 'z_39_a_104_liso_0_t', 'z_39_a_105_liso_0_t', 'z_39_a_90_liso_0', 'z_39_a_90_liso_1', 'z_39_a_91_liso_0', 'z_39_a_92_liso_0', 'z_39_a_93_liso_0', 'z_39_a_94_liso_0', 'z_39_a_95_liso_0', 'z_39_a_96_liso_0', 'z_39_a_96_liso_1', 'z_39_a_97_liso_0', 'z_39_a_97_liso_1', 'z_39_a_97_liso_2', 'z_39_a_98_liso_0', 'z_39_a_98_liso_1', 'z_39_a_99_liso_0', 'z_3_a_8_liso_0', 'z_3_a_9_liso_0', 'z_40_a_100_liso_0', 'z_40_a_101_liso_0', 'z_40_a_102_liso_0', 'z_40_a_103_liso_0_t', 'z_40_a_104_liso_0_t', 'z_40_a_105_liso_0_t', 'z_40_a_106_liso_0_t', 'z_40_a_107_liso_0_t', 'z_40_a_108_liso_0_t', 'z_40_a_93_liso_0', 'z_40_a_95_liso_0', 'z_40_a_97_liso_0', 'z_40_a_98_liso_0_t', 'z_40_a_99_liso_0', 'z_41_a_100_liso_0', 'z_41_a_100_liso_1', 'z_41_a_101_liso_0', 'z_41_a_102_liso_0', 'z_41_a_102_liso_1', 'z_41_a_103_liso_0', 'z_41_a_104_liso_0_t', 'z_41_a_104_liso_1', 'z_41_a_105_liso_0_t', 'z_41_a_106_liso_0', 'z_41_a_107_liso_0_t', 'z_41_a_108_liso_0_t', 'z_41_a_109_liso_0_t', 'z_41_a_110_liso_0_t', 'z_41_a_95_liso_0', 'z_41_a_95_liso_1', 'z_41_a_96_liso_0', 'z_41_a_97_liso_0', 'z_41_a_98_liso_0', 'z_41_a_98_liso_1', 'z_41_a_99_liso_0', 'z_41_a_99_liso_1', 'z_42_a_101_liso_0', 'z_42_a_102_liso_0', 'z_42_a_103_liso_0', 'z_42_a_104_liso_0', 'z_42_a_105_liso_0', 'z_42_a_106_liso_0_t', 'z_42_a_107_liso_0_t', 'z_42_a_108_liso_0_t', 'z_42_a_109_liso_0_t', 'z_42_a_110_liso_0', 'z_42_a_111_liso_0_t', 'z_42_a_112_liso_0_t', 'z_42_a_113_liso_0_t', 'z_42_a_99_liso_0', 'z_43_a_100_liso_0', 'z_43_a_101_liso_0', 'z_43_a_102_liso_0', 'z_43_a_102_liso_1', 'z_43_a_103_liso_0', 'z_43_a_104_liso_0', 'z_43_a_105_liso_0', 'z_43_a_106_liso_0', 'z_43_a_107_liso_0', 'z_43_a_108_liso_0_t', 'z_43_a_109_liso_0_t', 'z_43_a_110_liso_0_t', 'z_43_a_111_liso_0_t', 'z_43_a_112_liso_0_t', 'z_43_a_113_liso_0_t', 'z_43_a_114_liso_0_t', 'z_43_a_115_liso_0_t', 'z_43_a_99_liso_0', 'z_43_a_99_liso_1', 'z_44_a_103_liso_0', 'z_44_a_105_liso_0', 'z_44_a_106_liso_0', 'z_44_a_107_liso_0', 'z_44_a_108_liso_0', 'z_44_a_109_liso_0', 'z_44_a_110_liso_0', 'z_44_a_111_liso_0_t', 'z_44_a_112_liso_0_t', 'z_44_a_113_liso_0', 'z_44_a_114_liso_0_t', 'z_44_a_115_liso_0_t', 'z_44_a_116_liso_0_t', 'z_44_a_117_liso_0_t', 'z_44_a_118_liso_0_t', 'z_44_a_119_liso_0_t', 'z_44_a_120_liso_0_t', 'z_45_a_105_liso_0', 'z_45_a_106_liso_0', 'z_45_a_106_liso_1', 'z_45_a_107_liso_0', 'z_45_a_108_liso_0', 'z_45_a_109_liso_0', 'z_45_a_110_liso_0', 'z_45_a_110_liso_1', 'z_45_a_111_liso_0', 'z_45_a_112_liso_0_t', 'z_45_a_112_liso_1', 'z_45_a_113_liso_0_t', 'z_45_a_114_liso_0_t', 'z_45_a_115_liso_0_t', 'z_45_a_116_liso_0_t', 'z_45_a_116_liso_1', 'z_45_a_117_liso_0_t', 'z_45_a_118_liso_0_t', 'z_45_a_119_liso_0_t', 'z_45_a_120_liso_0_t', 'z_45_a_121_liso_0_t', 'z_45_a_122_liso_0_t', 'z_46_a_107_liso_0', 'z_46_a_109_liso_0', 'z_46_a_111_liso_0', 'z_46_a_111_liso_1', 'z_46_a_112_liso_0', 'z_46_a_113_liso_0', 'z_46_a_114_liso_0', 'z_46_a_115_liso_0_t', 'z_46_a_116_liso_0', 'z_46_a_117_liso_0_t', 'z_46_a_118_liso_0', 'z_46_a_119_liso_0_t', 'z_46_a_120_liso_0_t', 'z_46_a_121_liso_0_t', 'z_46_a_122_liso_0_t', 'z_46_a_123_liso_0_t', 'z_46_a_124_liso_0_t', 'z_47_a_110_liso_0', 'z_47_a_110_liso_1', 'z_47_a_111_liso_0', 'z_47_a_111_liso_1', 'z_47_a_112_liso_0', 'z_47_a_113_liso_0', 'z_47_a_113_liso_1', 'z_47_a_114_liso_0', 'z_47_a_115_liso_0', 'z_47_a_115_liso_1', 'z_47_a_116_liso_0', 'z_47_a_116_liso_1', 'z_47_a_117_liso_0', 'z_47_a_117_liso_1', 'z_47_a_118_liso_0_t', 'z_47_a_118_liso_1', 'z_47_a_119_liso_0', 'z_47_a_120_liso_0_t', 'z_47_a_120_liso_1', 'z_47_a_121_liso_0', 'z_47_a_122_liso_0', 'z_47_a_122_liso_1_t', 'z_47_a_123_liso_0_t', 'z_47_a_124_liso_0_t', 'z_47_a_125_liso_0_t', 'z_47_a_126_liso_0_t', 'z_47_a_127_liso_0_t', 'z_47_a_128_liso_0_t', 'z_47_a_129_liso_0_t', 'z_48_a_113_liso_0', 'z_48_a_113_liso_1', 'z_48_a_115_liso_0', 'z_48_a_115_liso_1', 'z_48_a_117_liso_0', 'z_48_a_117_liso_1', 'z_48_a_118_liso_0', 'z_48_a_119_liso_0', 'z_48_a_119_liso_1', 'z_48_a_120_liso_0', 'z_48_a_121_liso_0', 'z_48_a_121_liso_1', 'z_48_a_122_liso_0', 'z_48_a_123_liso_0', 'z_48_a_123_liso_1', 'z_48_a_124_liso_0_t', 'z_48_a_125_liso_0', 'z_48_a_125_liso_1', 'z_48_a_126_liso_0_t', 'z_48_a_127_liso_0_t', 'z_48_a_128_liso_0_t', 'z_48_a_129_liso_0_t', 'z_48_a_130_liso_0_t', 'z_48_a_131_liso_0_t', 'z_49_a_115_liso_0', 'z_49_a_115_liso_1', 'z_49_a_116_liso_0', 'z_49_a_116_liso_1', 'z_49_a_117_liso_0', 'z_49_a_117_liso_1', 'z_49_a_118_liso_0', 'z_49_a_118_liso_1', 'z_49_a_118_liso_2', 'z_49_a_119_liso_0', 'z_49_a_119_liso_1', 'z_49_a_120_liso_0', 'z_49_a_120_liso_1', 'z_49_a_120_liso_2', 'z_49_a_121_liso_0', 'z_49_a_121_liso_1', 'z_49_a_122_liso_0', 'z_49_a_122_liso_1', 'z_49_a_122_liso_2', 'z_49_a_123_liso_0', 'z_49_a_123_liso_1', 'z_49_a_124_liso_0', 'z_49_a_124_liso_1', 'z_49_a_125_liso_0', 'z_49_a_125_liso_1', 'z_49_a_126_liso_0', 'z_49_a_126_liso_1', 'z_49_a_127_liso_0', 'z_49_a_127_liso_1', 'z_49_a_128_liso_0', 'z_49_a_128_liso_1', 'z_49_a_129_liso_0', 'z_49_a_129_liso_1', 'z_49_a_130_liso_0', 'z_49_a_130_liso_1', 'z_49_a_130_liso_2', 'z_49_a_131_liso_0_t', 'z_49_a_131_liso_1_t', 'z_49_a_131_liso_2_t', 'z_49_a_132_liso_0', 'z_49_a_133_liso_0_t', 'z_49_a_133_liso_1_t', 'z_49_a_134_liso_0_t', 'z_4_a_10_liso_0', 'z_4_a_12_liso_0', 'z_50_a_121_liso_0', 'z_50_a_121_liso_1', 'z_50_a_123_liso_0', 'z_50_a_123_liso_1', 'z_50_a_125_liso_0', 'z_50_a_125_liso_1', 'z_50_a_126_liso_0', 'z_50_a_127_liso_0', 'z_50_a_127_liso_1', 'z_50_a_128_liso_0', 'z_50_a_129_liso_0', 'z_50_a_129_liso_1', 'z_50_a_130_liso_0', 'z_50_a_130_liso_1', 'z_50_a_131_liso_0_t', 'z_50_a_132_liso_0', 'z_50_a_133_liso_0', 'z_50_a_134_liso_0', 'z_50_a_135_liso_0_t', 'z_50_a_136_liso_0_t', 'z_51_a_124_liso_0', 'z_51_a_124_liso_1', 'z_51_a_125_liso_0', 'z_51_a_126_liso_0', 'z_51_a_126_liso_1', 'z_51_a_127_liso_0', 'z_51_a_128_liso_0', 'z_51_a_128_liso_1', 'z_51_a_129_liso_0', 'z_51_a_130_liso_0', 'z_51_a_130_liso_1', 'z_51_a_131_liso_0', 'z_51_a_132_liso_0', 'z_51_a_132_liso_1', 'z_51_a_133_liso_0', 'z_51_a_134_liso_0', 'z_51_a_134_liso_1', 'z_51_a_135_liso_0', 'z_51_a_136_liso_0_t', 'z_51_a_137_liso_0_t', 'z_51_a_138_liso_0_t', 'z_51_a_139_liso_0_t', 'z_52_a_127_liso_0', 'z_52_a_127_liso_1', 'z_52_a_129_liso_0', 'z_52_a_129_liso_1', 'z_52_a_131_liso_0', 'z_52_a_131_liso_1', 'z_52_a_132_liso_0', 'z_52_a_133_liso_0', 'z_52_a_133_liso_1', 'z_52_a_134_liso_0', 'z_52_a_135_liso_0', 'z_52_a_136_liso_0', 'z_52_a_137_liso_0_t', 'z_52_a_138_liso_0_t', 'z_52_a_139_liso_0_t', 'z_52_a_140_liso_0_t', 'z_52_a_141_liso_0_t', 'z_52_a_142_liso_0_t', 'z_53_a_129_liso_0', 'z_53_a_130_liso_0', 'z_53_a_130_liso_1', 'z_53_a_131_liso_0', 'z_53_a_132_liso_0', 'z_53_a_132_liso_1', 'z_53_a_133_liso_0', 'z_53_a_134_liso_0', 'z_53_a_134_liso_1', 'z_53_a_135_liso_0', 'z_53_a_136_liso_0', 'z_53_a_136_liso_1', 'z_53_a_137_liso_0', 'z_53_a_138_liso_0', 'z_53_a_139_liso_0_t', 'z_53_a_140_liso_0_t', 'z_53_a_141_liso_0_t', 'z_53_a_142_liso_0_t', 'z_53_a_143_liso_0_t', 'z_53_a_144_liso_0_t', 'z_54_a_133_liso_0', 'z_54_a_135_liso_0', 'z_54_a_135_liso_1', 'z_54_a_137_liso_0', 'z_54_a_138_liso_0', 'z_54_a_139_liso_0', 'z_54_a_140_liso_0', 'z_54_a_141_liso_0', 'z_54_a_142_liso_0_t', 'z_54_a_143_liso_0_t', 'z_54_a_144_liso_0_t', 'z_54_a_145_liso_0_t', 'z_54_a_146_liso_0_t', 'z_54_a_147_liso_0_t', 'z_55_a_134_liso_0', 'z_55_a_135_liso_0', 'z_55_a_136_liso_0', 'z_55_a_137_liso_0', 'z_55_a_138_liso_0', 'z_55_a_138_liso_1', 'z_55_a_139_liso_0', 'z_55_a_140_liso_0', 'z_55_a_141_liso_0', 'z_55_a_142_liso_0', 'z_55_a_143_liso_0', 'z_55_a_144_liso_0_t', 'z_55_a_145_liso_0', 'z_55_a_146_liso_0_t', 'z_55_a_147_liso_0_t', 'z_55_a_148_liso_0_t', 'z_55_a_149_liso_0_t', 'z_56_a_139_liso_0', 'z_56_a_140_liso_0', 'z_56_a_141_liso_0', 'z_56_a_142_liso_0', 'z_56_a_143_liso_0', 'z_56_a_144_liso_0_t', 'z_56_a_145_liso_0', 'z_56_a_146_liso_0_t', 'z_56_a_147_liso_0', 'z_56_a_148_liso_0_t', 'z_56_a_149_liso_0_t', 'z_56_a_150_liso_0_t', 'z_56_a_151_liso_0_t', 'z_56_a_152_liso_0_t', 'z_57_a_140_liso_0', 'z_57_a_141_liso_0', 'z_57_a_142_liso_0', 'z_57_a_143_liso_0', 'z_57_a_144_liso_0', 'z_57_a_145_liso_0', 'z_57_a_146_liso_0', 'z_57_a_147_liso_0', 'z_57_a_148_liso_0', 'z_57_a_149_liso_0_t', 'z_57_a_150_liso_0_t', 'z_57_a_151_liso_0_t', 'z_57_a_152_liso_0_t', 'z_57_a_153_liso_0_t', 'z_57_a_154_liso_0_t', 'z_58_a_141_liso_0', 'z_58_a_143_liso_0', 'z_58_a_144_liso_0', 'z_58_a_145_liso_0', 'z_58_a_146_liso_0', 'z_58_a_147_liso_0', 'z_58_a_148_liso_0_t', 'z_58_a_149_liso_0_t', 'z_58_a_150_liso_0_t', 'z_58_a_151_liso_0_t', 'z_58_a_152_liso_0_t', 'z_58_a_153_liso_0_t', 'z_58_a_154_liso_0_t', 'z_58_a_155_liso_0_t', 'z_58_a_156_liso_0_t', 'z_58_a_157_liso_0_t', 'z_59_a_143_liso_0', 'z_59_a_144_liso_0', 'z_59_a_144_liso_1', 'z_59_a_145_liso_0', 'z_59_a_146_liso_0', 'z_59_a_147_liso_0', 'z_59_a_148_liso_0', 'z_59_a_148_liso_1', 'z_59_a_149_liso_0', 'z_59_a_150_liso_0', 'z_59_a_151_liso_0', 'z_59_a_152_liso_0', 'z_59_a_153_liso_0_t', 'z_59_a_154_liso_0_t', 'z_59_a_155_liso_0_t', 'z_59_a_156_liso_0_t', 'z_59_a_157_liso_0_t', 'z_59_a_158_liso_0_t', 'z_59_a_159_liso_0_t', 'z_5_a_12_liso_0', 'z_60_a_147_liso_0', 'z_60_a_149_liso_0', 'z_60_a_151_liso_0', 'z_60_a_152_liso_0', 'z_60_a_153_liso_0', 'z_60_a_154_liso_0', 'z_60_a_155_liso_0_t', 'z_60_a_156_liso_0_t', 'z_60_a_157_liso_0_t', 'z_60_a_158_liso_0_t', 'z_60_a_159_liso_0_t', 'z_60_a_160_liso_0_t', 'z_61_a_147_liso_0', 'z_61_a_149_liso_0', 'z_61_a_150_liso_0', 'z_61_a_151_liso_0', 'z_61_a_152_liso_0', 'z_61_a_152_liso_1', 'z_61_a_153_liso_0', 'z_61_a_154_liso_0', 'z_61_a_154_liso_1', 'z_61_a_155_liso_0_t', 'z_61_a_156_liso_0', 'z_61_a_157_liso_0_t', 'z_61_a_158_liso_0_t', 'z_61_a_159_liso_0_t', 'z_61_a_160_liso_0_t', 'z_61_a_161_liso_0_t', 'z_61_a_162_liso_0_t', 'z_62_a_151_liso_0', 'z_62_a_153_liso_0', 'z_62_a_155_liso_0', 'z_62_a_156_liso_0', 'z_62_a_157_liso_0', 'z_62_a_158_liso_0_t', 'z_62_a_159_liso_0', 'z_62_a_160_liso_0_t', 'z_62_a_161_liso_0_t', 'z_62_a_162_liso_0_t', 'z_62_a_163_liso_0_t', 'z_62_a_164_liso_0_t', 'z_62_a_165_liso_0_t', 'z_63_a_154_liso_0', 'z_63_a_155_liso_0', 'z_63_a_156_liso_0', 'z_63_a_157_liso_0', 
        'z_63_a_158_liso_0', 'z_63_a_159_liso_0', 'z_63_a_161_liso_0_t', 'z_63_a_162_liso_0_t', 'z_63_a_163_liso_0_t', 'z_63_a_164_liso_0_t', 'z_63_a_165_liso_0_t', 'z_64_a_159_liso_0', 'z_64_a_161_liso_0', 'z_64_a_162_liso_0', 'z_64_a_163_liso_0_t', 'z_64_a_164_liso_0_t', 'z_64_a_165_liso_0_t', 'z_65_a_160_liso_0', 'z_65_a_161_liso_0', 'z_65_a_162_liso_0', 'z_65_a_163_liso_0', 'z_65_a_164_liso_0', 'z_65_a_165_liso_0', 'z_66_a_165_liso_0', 'z_66_a_165_liso_1', 'z_66_a_166_liso_0', 'z_66_a_167_liso_0', 'z_66_a_168_liso_0', 'z_66_a_169_liso_0', 'z_67_a_166_liso_0', 'z_67_a_166_liso_1', 'z_67_a_167_liso_0', 'z_67_a_168_liso_0', 'z_67_a_170_liso_0', 'z_67_a_170_liso_1', 'z_68_a_169_liso_0', 'z_68_a_171_liso_0', 'z_68_a_172_liso_0', 'z_68_a_173_liso_0', 'z_69_a_171_liso_0', 'z_69_a_172_liso_0', 'z_69_a_173_liso_0', 'z_69_a_174_liso_0', 'z_6_a_14_liso_0', 'z_6_a_15_liso_0', 'z_70_a_175_liso_0', 'z_70_a_177_liso_0', 'z_71_a_177_liso_0']
    
    def __init__(self, dict, ifSimp=False):
        super().__init__(dict, ifSimp=ifSimp)
        self.Z=dict["Z"]
        self.A=dict["A"]
        self.liso=dict["liso"]
        self.epE=dict["epE"]
        self.ifT=dict["ifT"]
        self.name=dict["name"]
        self.disc=dict["disc"]
        self.binEdge=dict["E"]        
    
    @staticmethod
    def getVals(fileName, ifSum=False, discSlope=5):
        """
        Reads from nuclear spectra file and obtains all important info about sample

        Args:
            fileName (str): file path to access data 
            ifSum (bool, optional): Denotes if string being read is a sum spectra 
                (less information is outputted; it's just set to 0). Defaults to False.
            discSlope (float, optional): sets the slope required to count a jump as a discontinuity
                Defaults to 5

        Returns:
            dictionary: contains: "Z", "A", "ifT" (if theoretical), "liso", "epE" (Endpoint energy), "name", 
                "E"(energy), "S"(spectrum), "disc" (array of tuples containing jump discontinuities as well)
                Note: "epE" is the energy of the last entry in "disc"
        """
        data=""
        dict={}
        with open(fileName,"r") as f:
            data = f.read()
        strArr=Util.strToStrArr(data, delimeter=";", numCols=2)
        dict["E"], dict["S"]=np.zeros(len(strArr[0])), np.zeros(len(strArr[0]))
        for i in range(len(dict["E"])):
            dict["E"][i]=float(strArr[0][i])
            dict["S"][i]=float(strArr[1][i])
        
        if ifSum:
            dict["Z"]=0
            dict["A"]=0
            dict["ifT"]=False
            dict["liso"]=0
            dict["epE"]=0
            dict["name"]="sum"
            return dict
        
        info=fileName[fileName.index("z"):]
        und=list(re.finditer(r"_", info))
        dict["Z"]=int(info[und[0].span()[0]+1: und[1].span()[0]])
        dict["A"]=int(info[und[2].span()[0]+1: und[3].span()[0]])
        if len(und)>5:
            dict["ifT"]=True
            dict["liso"]=int(info[und[4].span()[0]+1:und[5].span()[0]])
        else:
            dict["ifT"]=False
            dict["liso"]=int(info[und[4].span()[0]+1:info.index(".")])
        dict["epE"]=dict["E"][-1]
        dict["name"]=info[:info.index(".")]
        
        tempE=dict["E"]
        tempS=dict["S"]
        disc=[]
        for x in range(len(tempE)-1):
            if abs((tempS[x+1]-tempS[x])/(tempE[x+1]-tempE[x]))>discSlope:
                disc.append((tempE[x], abs(tempS[x+1]-tempS[x])))
        dict["disc"]=disc
        return dict
    
    @staticmethod
    def makeDatabase(endFileName, startingFolder=r"Nuclide Spectral Data\specific nuclides\z*.txt"):
        """
        Converts folder of nuclide data into a CSV database of dictionaries
            Allows for easy conversion to nuclideSpectra objects
        
        Parameters:
            endFileName (string): the name for the csv file created
            startingFolder (string, optional): the file path and command for the glob read. Defaults to
                standard value used in current codewriting
        
        Returns:
            Nothing (but saves the csv file in location specified (defaults to Workspace))
        """
        nuclides=glob.glob(startingFolder)
        dict={}
        for fileName in nuclides:
            obj=nuclideSpectra(nuclideSpectra.getVals(fileName))
            dict[obj.name]=obj.__dict__
        with open(endFileName, 'w') as csv_file:  
            writer = csv.writer(csv_file)
            for key, value in dict.items():
                writer.writerow([key, value])

    @staticmethod
    def makeReactDatabase(startingFolder, endFile, percents, ifSum=False, ifCS=True):
        """
        Creates an associated CSV file of individual nuclides for an arbitrary reactor (with arbit Fission Yields)
        
        Args:
            startingFolder (string): the location of all the raw data for individual unscaled nuclear spectra
            endFile (string): the location to save the final csv file, including the extension .csv
            percents (dictionary of 4 floats): the percentage of u5, u8, pu9, pu1 in the reactor. 
                Use keys of form 'u5','u8', 'pu9', 'pu1' in the dictionary
            ifSum (boolean, optional): if True, just creates a file summation instead of the nuclear database
                Defaults to False
            ifCS (boolean, optional): if True, multiplies spectra by the IBD cross section. 
                Defaults to True
            
        Returns:
            Nothing, but saves the dictionary at endFile 
        """
        count=0
        dictWeight={}
        for nucl in ['u5', 'u8', 'pu9', 'pu1']:
            fileName="Nuclide Spectral Data\\" +nucl+"cfy.txt"
            print(fileName)
            with open(fileName, 'r') as f:
                data=f.read()
            strArr=Util.strToStrArr(data, delimeter=";", numCols=4)
            for i in range(len(strArr[0])):
                strKey="z_"+str(strArr[0][i])+"_a_"+str(strArr[1][i])+"_liso_"+str(strArr[2][i])
                if strKey not in nuclideSpectra.names:
                    strKey+="_t"
                if count==0:
                    dictWeight[strKey]=float(strArr[3][i])*percents[nucl]
                else:
                    dictWeight[strKey]+=float(strArr[3][i])*percents[nucl]
            count+=1
        nuclides=glob.glob(startingFolder)
        dict={}

        if ifSum:
            x=""
            for fileName in nuclides:
                print(fileName)
                obj=nuclideSpectra(nuclideSpectra.getVals(fileName))
                dict[obj.name]=obj
                dict[obj.name].S=obj.S*dictWeight[obj.name]
                if x=="":
                    x=obj.name
            wSum=sumSpectra.sumNucl(dict, bins=dict[x].E)
            dict2={"Fine Sum":wSum}
            Util.dictToCSV(endFile, dict2)
        else:
            for fileName in nuclides:
                print(fileName)
                obj=nuclideSpectra(nuclideSpectra.getVals(fileName))
                dict[obj.name]=obj.__dict__
                if ifCS:
                    dict[obj.name]["S"]=obj.S[:len(sumSpectra.crossSection)]*dictWeight[obj.name]*sumSpectra.crossSection[:len(obj.S)]
                else:
                    dict[obj.name]["S"]=obj.S*dictWeight[obj.name]
                dict[obj.name]["E"]=obj.E[:len(obj.S)]
            Util.dictToCSV(endFile, dict)
    
    def getName(self):
        """Returns the name of the associated file of this nuclide"""
        if self.ifT:
            return "z_"+str(self.Z)+"_a_"+str(self.A)+"_liso_"+str(self.liso)+"_t.txt"
        return "z_"+str(self.Z)+"_a_"+str(self.A)+"_liso_"+str(self.liso)+".txt"
        
class redRelSpectra(Spectra):
    """
    Contains a reduced number of crucial terms for the spectra, of released data
    Used for released NEOS/RENO/Daya Bay data, to account for uncertainties and all that
    
    Attributes:
        delS (array of floats): uncertainty of S
        cov (2d array of floats): covariance matrix
        delR (array of floats): uncertainty in R
        
    Class Methods:
        calcUnc: calculates uncertainty
        Redefines rebin, renorm
    """
    def __init__(self, dict, ifSimp=False):
        if "unfE" in dict:
            super().__init__({"E": dict["unfE"], "S":dict['unfS']}, ifSimp=ifSimp)
            self.cov=dict["covUnf"]
            self.binEdge=dict["unfBins"]
            self.renorm()
            if not ifSimp:
                self.calcUnc()
        if "promptE" in dict:
            super().__init__({"E": dict["promptE"], "S": dict["promptS"]})
            self.cov=dict["covPrompt"]
            self.calcUnc()
            if "binEdge" in dict:
                self.binEdge=dict["binEdge"]
        else:
            super().__init__(dict)
            self.cov=dict["cov"]
            self.delR=dict["delR"]
            self.delS=dict["delS"]
            if "binEdge" in dict:
                self.binEdge=dict["binEdge"]

    def calcUnc(self):
        """Calculates delS and delR"""
        self.delS=np.zeros(len(self.E))
        self.delR=np.zeros(len(self.E))
        for x in range(len(self.E)):
            self.delS[x]=math.sqrt(self.cov[x][x])
        for x in range(len(self.R)):
            self.delR[x]=math.sqrt(abs(self.S[x+1]**(-2)*self.cov[x][x]+self.S[x]**2*self.S[x+1]**(-4)*
                                       self.cov[x+1][x+1]-2*self.S[x]*self.S[x+1]**(-3)*self.cov[x][x+1]))
    
    def rebin(self, newBins):
        """Rebins the data, and the covariance matrix too

        Args:
            newBins (arraylike of floats): indexes upon which the bins are being made in the energy data
                Don't include the last index, and that bin will just be the total at the end
        """
        binInd=self.getBinInd(newBins)
        lenE=len(self.E)
        super().rebin(newBins)
        self.E=self.E[:len(binInd)]
        self.S=self.S[:len(binInd)]
        self.R=self.R[:len(binInd)-1]
        self.E_avg=self.E_avg[:len(self.R)]
        newCov = np.zeros((len(binInd), len(binInd)))
        for i in range(len(binInd)-1):
            for j in range(i, len(binInd)-1): #Associates with a specific position in the new covariance matrix
                covVal=0
                num=0
                for a in range(binInd[i], binInd[i+1]):
                    for b in range(binInd[j], binInd[j+1]):
                        covVal+=self.cov[a][b]
                        num+=1
                newCov[i][j], newCov[j][i]=covVal/num, covVal/num
        for i in range(len(binInd)-1):
            covVal=0
            for a in range(binInd[-1], len(self.cov)):
                for b in range(binInd[i], binInd[i+1]):
                    covVal+=self.cov[a][b]
            newCov[i][-1], newCov[-1][i]=covVal, covVal
        covVal=0
        for a in range(binInd[-1], len(self.cov)):
            for b in range(binInd[-1], len(self.cov)):
                covVal+=self.cov[a][b]
        newCov[-1][-1]=covVal
        self.cov=newCov
        self.calcUnc()
        return self

    def renorm(self):
        """Renormalizes the spectra such that the integral underneath is 1

        Args:
            binSize (float): the width of each bin (assumes equal bin width)
        Returns:
            Nothing, but renormalizes the values (and covariance)
        """
        count=super().renorm()  
        self.cov=self.cov/(count**2)
        self.calcUnc()
        return count
        
class releasedSpectra(Spectra):
    """Class containing released data from NEOS/RENO sources, and methods to analyze them
    
    Attributes:
        promptE (array of floats): energy levels for prompt spectrum
        promptS (array of floats): spectra values for prompt spectrum
        promptBins (array of floats): bin edges for prompt spectrum
        covPrompt (2D array of floats): covariance matrix for 2D array
        unfE (array of floats): energy levels for unfolded spectrum
        unfS (array of floats): spectra values for prompt spectrum
        unfBins (array of floats): bin values for prompt spectrum
        covUnf (2D array of floats): covariance matrix for 2D array
        respMat (2D array of floats): response matrix for reactor
        
    Class Methods:
        extractVals: used to extract data from official data releases from NEOS/RENO
    """

    def __init__(self, dict, ifSimp=False):
        self.promptE=dict["promptE"]
        self.promptS=dict["promptS"]
        self.promptBins=dict["promptBins"]
        self.covPrompt=dict["covPrompt"]
        self.unfE=dict["unfE"]
        self.unfS=dict["unfS"]
        self.unfBins=dict["unfS"]
        self.covUnf=dict["covUnf"]
        self.respMat=dict["respMat"]
        if not ifSimp:
            for x in ["prompt", "unf"]:
                strR=x+"R"
                strE_avg=x+"E_avg"
                if strR in dict:
                    self[strR]=dict[strR]
                    self[strE_avg]=dict[strE_avg]
                else:
                    strE=x+"E"
                    strS=x+"S"
                    dict=self.__dict__
                    for x in range(len(dict[strE])-1):
                        self.E_avg.append(.5*(dict[strE][x]+dict[strE][x+1]))
                        if (dict[strS][x+1]!=0):
                            self.R.append(dict[strS][x]/dict[strS][x+1])
                        else:
                            self.R.append(dict[strS][x]/1E-20)     
        
    @staticmethod
    def extractVals(filePath, respSize, dataOrder=['promptS', 'covPrompt', 'respMat', 'unfS', 'covUnf'], delims={'promptS': "\t", 'covPrompt': "\t", 'respMat': "\t", 'unfS': "\t", 'covUnf': " "}, ifDaya=False):
        """
        Extracts information from the filePath related to the released spectra, encapsulating it in dictionary form
            Current possible dictionary values:
                promptS: includes promptE, promptS, promptBins
                covPrompt: adds covariance matrix for Prompt data
                respMat: adds response matrix
                unfS: adds unfolded data (unfE, unfS, unfBins)
                covUnf: adds covariance matrix for Unfolded data
                
        #*****# Specifically for the formatting presented in RENO/NEOS data (changes for Daya Bay data saving)
        
        Args:
            filePath (string): path to open text file containing data
            respSize (arraylike with 2 floats): contains [row, height] information for repsonse matrix
            dataOrder (list of strings, optional): contains order and dictionary values present in the file to be read.
                Defaults to ['promptS', 'covPrompt', 'respMat', 'unfS', 'covUnf'].
            delims (dictionary): contains the name for each data entry, keying to the specific delimeter used to separate different elements in that array
            ifDaya (boolean, optional): if the data is in Daya Bay format. Defaults to False
        
        Returns:
            dictionary: contains the specific values specified in the dataOrder array
        """
        data=""
        dict={}
        count=0
        with open(filePath,"r") as f:
            data = f.read()
        lines=data.splitlines(keepends=True)
        indDash=[]
        for x in range(len(lines)):
            if lines[x].__contains__("---"):
                indDash.append(x)
        ifOdd=False
        for i in range(len(indDash)):
            if ifOdd:
                ifOdd=False
                if i==len(indDash)-1:
                    lineSet=lines[indDash[i]+1:]
                else:
                    lineSet=lines[indDash[i]+1:indDash[i+1]]
                dForm=dataOrder[count]
                count+=1
                match dForm:
                    case 'promptS'|'unfS':
                        if ifDaya:
                            strArr=Util.strToStrArr(lineSet, delims[dForm], 3)
                            dict['promptE']=[float(x) for x in strArr[0]]
                            dict['promptS']=[float(x) for x in strArr[1]]                            
                        else:
                            strArr=Util.strToStrArr(lineSet, delims[dForm], 4)
                            dict[dForm[:-1]+"E"]=[float(x) for x in strArr[1]]
                            dict[dForm]=[float(x) for x in strArr[3]]
                            dict[dForm[:-1]+"Bins"]=[float(x) for x in strArr[0]]
                            dict[dForm[:-1]+"Bins"].append(float(strArr[2][-1]))
                    case 'covPrompt' | 'covUnf'| 'respMat':
                        str=""
                        if dForm== 'covPrompt':
                            n=len(dict["promptE"])
                            m=n
                        elif dForm == 'covUnf':
                            n=len(dict["unfE"])
                            m=n
                        else:
                            n=respSize[0]
                            m=respSize[1]
                        lineLen=[]
                        for line in lineSet:
                            if line[-1:]=="\n":
                                str+=line[:-1]
                            else:
                                str+=line
                            lineLen.append(len(line))
                        arr=[]
                        for i in range(n):
                            arr.append([])
                        row, col=0, 0
                        cutLength=0
                        while (row<n)| (col<m):
                            if len(str)==0:
                                break
                            try:
                                backInd=str.index(delims[dForm])
                            except ValueError:
                                arr[row].append(float(str))
                                row, col= n, m
                                break
                            try: 
                                float(str[0:backInd])
                            except ValueError:
                                for i in range(len(lineSet)):
                                    temp=""
                                    if lines[i][-1:]=="\n":
                                        temp=lineSet[i][:-1]+lineSet[i+1]
                                    else:
                                        temp=lineSet[i]+lineSet[i+1]
                                    if str[0:backInd] in temp:
                                        lastTab=lineSet[i].rfind(delims[dForm])
                                        arr[row].append(float(lineSet[i][lastTab+1:]))
                                        str=str[lineLen[i]-lastTab-2:]
                                        break
                            else:
                                arr[row].append(float(str[0:backInd]))
                                str=str[backInd+1:]
                            if col<n-1:
                                col+=1
                            else:
                                col=0
                                row+=1
                        dict[dForm]=arr
            else:
                ifOdd=True
        return dict

class HuberMueller:
    """
    Class to deal with Huber Mueller models and data
    
    Class Attributes:
        bins (2d array of floats): the standard binnings used for Huber Mueller data, from .1 to .8 MeV size bins

    Attributes:
        u5, u8, pu9, pu1 (nested lists of arrays): each contain the Huber Mueller model binned according 
            to class attribute bins
            
    Class Methods:
        convToSum: creates a sumSpectra object according to reactor fission yields
        getHMCoeffs: reads Huber Mueller model values from file
        getFine: reads and sums Huber Mueller model according to fine data (read from file)
        
    """
    #Bins all go from 1.8 to around 8 MeV. NOTE: size .1, .2, .3, .4, .5, .6, .7, .8 MeV bins
    bins=[[ 1.8, 1.9, 2., 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3., 3.1, 3.2, 3.3, 3.4, 3.5,
            3.6, 3.7, 3.8, 3.9, 4., 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9, 5., 5.1, 5.2, 5.3,
            5.4, 5.5, 5.6, 5.7, 5.8, 5.9, 6., 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 7., 7.1,
            7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9, 8. ],
            [1.8, 2., 2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.6, 3.8, 4., 4.2, 4.4, 4.6, 4.8, 5., 5.2,
            5.4, 5.6, 5.8, 6., 6.2, 6.4, 6.6, 6.8, 7., 7.2, 7.4, 7.6, 7.8, 8. ],
            [1.8, 2.1, 2.4, 2.7, 3., 3.3, 3.6, 3.9, 4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6., 6.3, 6.6, 6.9,
            7.2, 7.5, 7.8, 8.1 ],
            [ 1.8, 2.2, 2.6, 3., 3.4, 3.8, 4.2, 4.6, 5., 5.4, 5.8, 6.2, 6.6, 7., 7.4, 7.8, 8.2 ],
            [ 1.8, 2.3, 2.8, 3.3, 3.8, 4.3, 4.8, 5.3, 5.8, 6.3, 6.8, 7.3, 7.8, 8.3 ],
            [ 1.8, 2.4, 3., 3.6, 4.2, 4.8, 5.4, 6., 6.6, 7.2, 7.8, 8.4 ],
            [ 1.8, 2.5, 3.2, 3.9, 4.6, 5.3, 6., 6.7, 7.4, 8.1 ],
            [ 1.8, 2.6, 3.4, 4.2, 5., 5.8, 6.6, 7.4, 8.2 ]]
    
    def __init__(self):
        hmVals= HuberMueller.getHMCoeffs()
        self.u5=hmVals[0]
        self.u8=hmVals[1]
        self.pu9=hmVals[2]
        self.pu1=hmVals[3]
    
    @staticmethod
    def convToSum(percent):
        """
        Converts the percentages given above into a sumSpectra object

        Args:
            percent (dictionary containing floats): fission yield for each fuel nuclide

        Returns:
            (array of sumSpectra): associated sum spectra
        """
        hm=HuberMueller()
        arr=[]
        for x in range(8):
            Stot=np.array(hm.u5[x])*percent["u5"]
            for key in ["u8", "pu9", "pu1"]:
                Stot+=np.array(hm.__dict__[key][x])*percent[key]
            binEdge=np.array(HuberMueller.bins[x])
            E=np.zeros(len(binEdge)-1)
            for i in range(len(binEdge)-1):
                E[i]=.5*(binEdge[i]+binEdge[i+1])
            spect=sumSpectra({"E": E, "S": Stot, "binEdge":binEdge})
            spect.renorm()
            arr.append(spect)
        return arr     
                    
    @staticmethod
    def getHMCoeffs(fileName=r"READ FROM CODE\\HBBinned.csv"):
        """
        Reads the Huber Mueller coefficients off of file

        Args:
            fileName (regexp, optional): name/path of file to read from (in specific format written previously). Defaults to r"HBBinned.csv".

        Returns:
            (list of 4 nested lists): highest level is separation between 5, 8, 9, 1
                next level is between bin size (.1 -> .8, increasing by .1)
                lowest level is array containing spectral values
        """
        with open(fileName, "r") as f:
            data=f.read()
        lines=data.splitlines(keepends=True)
        indColon=[]
        for i in range(len(lines)):
            if lines[i].__contains__(":"):
                indColon.append(i)
        hmVals=[]
        for x in range(4):
            if x==3:
                lineSet=lines[indColon[x]+1:]
            else:
                lineSet=lines[indColon[x]+1:indColon[x+1]]
            str=""
            for line in lineSet:
                if line[-1]=="\n":
                    str+=line[:-1]
                else:
                    str+=line
            hmVals.append(eval(str))
        return hmVals
  
    @staticmethod
    def getFine(percent, fileName="HB fine.csv", bins=None):
        """
        Creates a spectra from the fine Huber Mueller data, according to a certain reactor percentage

        Args:
            percent (dictionary with 4 floats): the percentages of each fuel nuclide in the reactor.
                Keys: u5, u8, pu9, pu1 
            fileName (string, optional): file name to be drawn from; defaults to "HB fine.csv"
            bins(arraylike, optional): bins for the spectra. Defaults to Util.getBin default

        Returns:
            sumSpectra object: sumSpectra for the data
        """
        arr=[]
        with open(fileName, "r") as file:
            data=file.read()
            count=0
            for lines in data.splitlines():
                if count%2==1:
                    arr.append(eval(lines))
                count+=1
        bins=Util.getBin(bins)
        count=0
        spect=np.zeros(len(arr[0]))
        for x in ["u5", "u8", "pu9", "pu1"]:
            spect+=np.array(arr[count])*percent[x]
            count+=1
        hm=sumSpectra({"E":bins, "S":spect})
        hm.renorm()
        return hm
            
class BSTree:
    """
    Binary search tree class for easy searching of sorted data (specifically for fast computation for MC sims.)
    
    Attributes:
        value (float): value of the node
        index (int): the index of this value in the original array
        left (BSNode or None): the node to the left below current one (None if doesn't exist)
        right (BSNode or None): the node to the right below current one (None if doesn't exist)
    
    Class Methods:
        addNode: adds node to tree
        search: searches through tree (assumed to be ordered) for value
        makeTree: makes tree from array
        inOrder: returns in order traversal of tree (Left-Root-Right)
        preOrder: returns pre order traversal of tree (Root-Left-Right)
        treeFromFile: creates tree from file containing preOrder and inOrder info
        checkEq: checks equality of 2 trees
        subTreeCDF: subtracts one tree from arbitrary number of other trees
    """
    def __init__(self, val, index):
        self.value=val
        self.index=index
        self.left=None
        self.right=None

    def addNode(self, val, index):
        """
        Adds a node to the current tree by recursively iterating through the different levels
        
        Args:
            val (float): value to be added to the tree    
            index (int): index of the value in the original array, transferred down the chain to save it at the endpoint
        """ 
        if isinstance(self.value, type(None)):
            self.value=val
            return
        elif val<self.value:
            if isinstance(self.left, type(None)):
                self.left=BSTree(val, index)
                return
            self.left.addNode(val, index)
        else:
            if isinstance(self.right, type(None)):
                self.right=BSTree(val, index)
                return
            self.right.addNode(val, index)        
   
    def search(self, val):
        """
        Searches through a binary search tree in order to get the original index of that point in the array
            If value is not in tree, returns the last node it ends on 
            (if binning of distribution fine enough, should be close enough)
        
        Args:
            val (float): the value being searched for
        
        Returns:
            (int): the index of the element (if in the array) and of the last branch reached if not
        """
        if self.value==val:
            return self.index
        elif val<self.value:
            if isinstance(self.left, type(None)):
                return self.index
            return self.left.search(val)
        else:
            if isinstance(self.right, type(None)):
                return self.index
            return self.right.search(val)
    
    @staticmethod
    def makeTree(arr, rSeed=None):
        """
        Makes a tree from array, randomizing the order elements are selected in to ensure large tree depth
    
        Args:
            arr (arraylike of floats): array to make tree from
            rSeed (int, optional) the seed for the random number generation. Defaults to 2718
            
        Returns:
            (BSTree): Binary Search Tree
        """    
        inds=list(range(len(arr)))
        tree=BSTree(arr[int(len(arr)/2)], int(len(arr)/2))
        inds.pop(int(len(arr)/2))
        reps=len(arr)
        if not isinstance(seed, type(None)):
            seed(rSeed)
        else:
            seed(2718)
        for x in range(reps-1):
            i=randrange(0, len(inds))
            tree.addNode(arr[inds[i]], inds[i])
            inds.pop(i)
            print("Tree Node #:"+x)
        return tree
    
    def inOrder(self, arr=[[], []]):
        """
        Returns the in-order traversal for a BSTree (Left-Root-Right)
        
        Args:
            None
        
        Returns:
            (arraylike of 2 arrays): [arrVal, arrInd] both inOrder traversals
        """
        # if isinstance(self.left, type(None))& isinstance(self.right, type(None)):
        #     arr[0].append(self.value)
        #     arr[1].append(self.index)
        #     return arr
        if not isinstance(self.left, type(None)):
            arr=self.left.inOrder(arr)
        arr[0].append(float(self.value))
        arr[1].append(float(self.index))
        if not isinstance(self.right, type(None)):
            arr=self.right.inOrder(arr)
        return arr
    
    def preOrder(self, arr=[[], []]):
        """
        Returns the preorder traversal for a BSTree (Root-Left-Right)

        Args:
            None
            
        Returns:
            (arraylike containing 2 arrays): [arrVal, arrInd] both in the preorder traversal of the tree
        """
        
        arr[0].append(float(self.value))
        arr[1].append(float(self.index))
        if not isinstance(self.left, type(None)):
            arr=self.left.preOrder(arr)
        if not isinstance(self.right, type(None)):
            arr=self.right.preOrder(arr)
        return arr
    
    @staticmethod 
    def treeFromFile(fileName):
        """
        Constructs a tree in time O(n) from a file including preOrder and inOrder information
            Algorithm taken from https://www.geeksforgeeks.org/construct-tree-from-given-inorder-and-preorder-traversal/
                (last algorithm present)
                
        Args:
            fileName (string): file name for a csv containing 3 lines in the following order
                preOrder of data
                inOrder of data
                preOrder of indices
        
        Returns:
            (BSTree): the constructed tree
        """
        
        with open(fileName, "r") as file:
            data=file.read()
            lines=data.splitlines()
            preOrder=eval(lines[0])
            inOrder=eval(lines[1])
            preIndex=eval(lines[2])
        root=None
        iPre=0 #current index/node in the preOrder list
        iIn=0 #current index in the inOrder list
        stack=[] #set containing the nodes that we've already passed through on the preorder (that we will return to)
        s=set() #set indicating where to look for right nodes
        while iPre<len(preOrder):
            while True:
                node=BSTree(preOrder[iPre], preIndex[iPre])
                if isinstance(root, type(None)):
                    root=node
                if len(stack)>0:
                    if stack[-1] in s:#if there is element in stack, checks if we expect top stack element 
                        s.discard(stack[-1]) #to have node attached to its right
                        stack[-1].right=node # if so, attach it and remove from stack (and from list of nodes to consider for right end)
                        stack.pop()
                    else:
                        stack[-1].left=node #if node isn't attached to the right, goes on left
                stack.append(node)
                if (iPre>=len(preOrder)) or (preOrder[iPre]==inOrder[iIn]):
                    iPre+=1
                    break
                iPre+=1
            node=None
            while (len(stack)>0)and (iIn<len(preOrder)) and (stack[-1].value==inOrder[iIn]):
                node=stack[-1]
                stack.pop()
                iIn+=1
            if not isinstance(node, type(None)):
                s.add(node)
                stack.append(node)
        return root

    def checkEq(self, obj):
        """
        Checks equality between 2 tree objects (without redefining __eq__ to preserve __hash__)

        Args:
            obj (BSTree): tree to be compared to the current object

        Returns:
            boolean: returns if elements are equal or not
        """
        if (self.value==obj.value)& (self.index==obj.index):
            if (isinstance(self.left, type(None)) and isinstance(obj.left, type(None))) or (self.left.checkEq(obj.left)):
                if (isinstance(self.right, type(None)) and isinstance(obj.right, type(None))) or (self.right.checkEq(obj.right)):
                    return True
        return False                 

    def subTreeCDF(self, ifFirst=True, *args):
        """
        Subtracts the other trees from the main tree in O(n) time by recursively hitting each point
        
        Args:
            ifFirst (boolean, optional): ignore this, it just maintains that the code can differentiate between the first run and all others
            *args (arbitrary number of np array of float): array of float is an individual CDF, but properly scaled 
                according to its yield in the total distribution
                in all other runs, it will feed the total sum that is measured later
        """
        if ifFirst:
            arr=np.zeros(len(args[0]))
            for nucl in args:
                arr+=np.array(nucl)
            intScale=1-arr[-1]
        else:
            intScale=1-args[0][-1]
            arr=args[0]
        self.value=(self.value-arr[self.index])/(intScale) #args[0] is the reduced array, and [-1] represents last element
        if not isinstance(self.left, type(None)):
            self.left=self.left.subTreeCDF(False, arr)
        if not isinstance(self.right, type(None)):
            self.right=self.right.subTreeCDF(False, arr)
        return self
                                          
class dayaBay250(Spectra): 
    """Spectra for the Daya Bay data

    Attributes:
        delS: uncertainty in the spectrum (as a percentage of S)
        delR: uncertainty in the spectral ratio (absolute terms)
    
    Methods:
        dayaSpectra: Constructs dayaSpectra based on the available data
    """
    #actually the E_avg bins
    bins=[1.8, 2.05, 2.30, 2.55, 2.80, 3.05, 3.30, 3.55, 3.80, 4.05, 4.30, 4.55, 4.80, 5.05, 5.30, 5.55, 5.80, 6.05, 6.30, 6.55,
    6.80, 7.05, 7.5, 8, 8.5, 9, 9.5, 10, 10.5, 11]
    
    dayaData=["1.925 1.28793 11.54%",

"2.175 1.06046 2.69631%",

"2.425 0.877182 1.886%",

"2.675 0.735927 1.56739%",

"2.925 0.603947 1.41889%",

"3.175 0.511274 1.50251%",

"3.425 0.417965 1.69262%",

"3.675 0.336692 1.7148%",

"3.925 0.273716 1.78833%",

"4.175 0.217448 1.88763%",

"4.425 0.169012 1.84646%",

"4.675 0.128543 1.92192%",

"4.925 0.100837 2.1156%",

"5.175 0.0833094 2.01623%",

"5.425 0.0662842 1.77535%",

"5.675 0.0506637 2.09007%",

"5.925 0.0392418 2.06063%",

"6.175 0.0289769 2.20403%",

"6.425 0.020378 3.01311%",

"6.675 0.0150936 2.47672%",

"6.925 0.0107762 3.14995%",

"7.275 0.00607851 2.69578%",

"7.75 0.00248128 2.85352%",

"8.25 0.000726664 5.39083%",

"8.75 0.000159723 12.1643%",

"9.25 4.75865e-05 21.7178%",

"9.75 1.76574e-05 31.1431%",

"10.25 1.6312e-05 23.5644%",
"10.75 2.55633e-06 93.1814%"]
    
    dayaCov=[[0.0220901 ,-4.47085e-05 ,0.000105065 ,0.000273692 ,-4.34054e-05 ,-0.000188469 ,-0.000284638 ,
          -0.00024108 ,-0.000227356 ,-0.000223224 ,-0.000120696 ,-0.00012369 ,-0.000114365 ,-7.12598e-05 ,-3.00274e-05 ,-4.3572e-05 ,-3.34794e-05 ,-1.43785e-05 ,-3.26909e-05 ,-1.01833e-05 ,-4.34014e-06 ,2.49897e-07 ,3.73215e-10 ,-6.92071e-08 ,-1.15952e-08 ,-1.63972e-09 ,1.02472e-09 ,-4.93594e-11 ,-2.63688e-10],

[-4.47085e-05 ,0.000817582 ,0.000277441 ,0.000208814 ,0.000107997 ,6.42073e-05 ,4.92682e-05 ,2.73386e-05 ,2.39211e-05 ,1.76499e-05 ,1.67265e-05 ,9.31717e-06 ,8.76424e-06 ,1.39715e-05 ,7.13919e-06 ,4.97064e-06 ,5.44068e-06 ,3.12404e-06 ,3.75157e-06 ,2.68886e-06 ,1.24177e-06 ,3.85689e-07 ,1.0197e-07 ,-8.67471e-09 ,4.23436e-10 ,1.22621e-09 ,1.40207e-09 ,7.60943e-10 ,-1.78143e-11],

[0.000105065 ,0.000277441 ,0.000273693 ,0.000141343 ,0.000108101 ,9.03448e-05 ,8.14274e-05 ,5.61773e-05 ,4.42851e-05 ,3.91564e-05 ,2.90678e-05 ,1.99106e-05 ,1.79284e-05 ,1.7118e-05 ,1.03818e-05 ,8.73318e-06 ,6.8597e-06 ,4.82149e-06 ,5.26283e-06 ,2.1547e-06 ,1.68715e-06 ,5.26071e-07 ,1.96708e-07 ,4.48683e-08 ,1.11743e-08 ,3.80469e-09 ,1.76038e-09 ,1.47941e-09 ,1.95731e-10],

[0.000273692 ,0.000208814 ,0.000141343 ,0.000133053 ,6.85566e-05 ,6.21527e-05 ,5.18002e-05 ,3.77165e-05 ,2.97853e-05 ,2.15664e-05 ,1.83658e-05 ,1.37436e-05 ,1.05431e-05 ,1.00345e-05 ,7.41505e-06 ,5.35857e-06 ,4.08441e-06 ,3.88158e-06 ,2.62957e-06 ,1.30385e-06 ,1.51784e-06 ,5.23056e-07 ,2.0845e-07 ,5.72827e-08 ,1.34373e-08 ,4.23108e-09 ,1.69893e-09 ,1.55271e-09 ,2.4066e-10],

[-4.34054e-05 ,0.000107997 ,0.000108101 ,6.85566e-05 ,7.34337e-05 ,4.8413e-05 ,4.7436e-05 ,3.85662e-05 ,2.91331e-05 ,2.39835e-05 ,1.85952e-05 ,1.343e-05 ,1.13264e-05 ,1.02427e-05 ,5.84693e-06 ,5.60431e-06 ,4.65295e-06 ,2.74512e-06 ,2.75205e-06 ,1.64029e-06 ,1.09939e-06 ,4.56338e-07 ,1.84685e-07 ,5.29021e-08 ,1.22299e-08 ,3.78053e-09 ,1.459e-09 ,1.35956e-09 ,2.18137e-10],

[-0.000188469 ,6.42073e-05 ,9.03448e-05 ,6.21527e-05 ,4.8413e-05 ,5.90125e-05 ,4.50049e-05 ,3.57137e-05 ,3.06564e-05 ,2.56035e-05 ,1.84038e-05 ,1.45763e-05 ,1.17234e-05 ,9.28979e-06 ,6.64585e-06 ,5.92297e-06 ,3.90338e-06 ,3.04697e-06 ,2.98113e-06 ,1.1969e-06 ,1.09462e-06 ,3.95721e-07 ,1.60253e-07 ,4.61024e-08 ,1.05885e-08 ,3.25238e-09 ,1.24513e-09 ,1.15929e-09 ,1.85831e-10],

[-0.000284638 ,4.92682e-05 ,8.14274e-05 ,5.18002e-05 ,4.7436e-05 ,4.50049e-05 ,5.00493e-05 ,3.42135e-05 ,3.00565e-05 ,2.50346e-05 ,1.90401e-05 ,1.40244e-05 ,1.17707e-05 ,9.62616e-06 ,6.19094e-06 ,5.66206e-06 ,3.92355e-06 ,3.03936e-06 ,3.07196e-06 ,1.09983e-06 ,1.05421e-06 ,3.4493e-07 ,1.40514e-07 ,4.11047e-08 ,9.23851e-09 ,2.78634e-09 ,1.04283e-09 ,9.71687e-10 ,1.55553e-10],

[-0.00024108 ,2.73386e-05 ,5.61773e-05 ,3.77165e-05 ,3.85662e-05 ,3.57137e-05 ,3.42135e-05 ,3.33343e-05 ,2.28983e-05 ,2.05546e-05 ,1.50132e-05 ,1.17707e-05 ,9.89392e-06 ,7.91889e-06 ,4.80733e-06 ,4.64559e-06 ,3.57936e-06 ,2.42184e-06 ,2.33168e-06 ,1.06134e-06 ,9.1409e-07 ,2.90174e-07 ,1.1824e-07 ,3.47436e-08 ,7.68422e-09 ,2.29329e-09 ,8.48471e-10 ,7.84871e-10 ,1.24214e-10],

[-0.000227356 ,2.39211e-05 ,4.42851e-05 ,2.97853e-05 ,2.91331e-05 ,3.06564e-05 ,3.00565e-05 ,2.28983e-05 ,2.39604e-05 ,1.65874e-05 ,1.27683e-05 ,1.04726e-05 ,9.0174e-06 ,6.44473e-06 ,4.16883e-06 ,4.31131e-06 ,2.81118e-06 ,2.0481e-06 ,2.14333e-06 ,8.63974e-07 ,7.63118e-07 ,2.38877e-07 ,9.68253e-08 ,2.80824e-08 ,6.1508e-09 ,1.82504e-09 ,6.80386e-10 ,6.20921e-10 ,9.55121e-11],

[-0.000223224 ,1.76499e-05 ,3.91564e-05 ,2.15664e-05 ,2.39835e-05 ,2.56035e-05 ,2.50346e-05 ,2.05546e-05 ,1.65874e-05 ,1.68477e-05 ,1.04574e-05 ,8.71363e-06 ,7.25303e-06 ,5.66281e-06 ,3.63137e-06 ,3.39338e-06 ,2.41228e-06 ,1.59998e-06 ,1.86108e-06 ,7.30629e-07 ,5.66107e-07 ,2.03687e-07 ,8.32451e-08 ,2.46654e-08 ,5.28874e-09 ,1.54136e-09 ,5.58364e-10 ,5.11259e-10 ,7.89289e-11],

[-0.000120696 ,1.67265e-05 ,2.90678e-05 ,1.83658e-05 ,1.85952e-05 ,1.84038e-05 ,1.90401e-05 ,1.50132e-05 ,1.27683e-05 ,1.04574e-05 ,9.73909e-06 ,5.71978e-06 ,5.24238e-06 ,4.42283e-06 ,2.74234e-06 ,2.49661e-06 ,1.71911e-06 ,1.32059e-06 ,1.27829e-06 ,5.83616e-07 ,4.41204e-07 ,1.50204e-07 ,6.03985e-08 ,1.72088e-08 ,3.71628e-09 ,1.09564e-09 ,4.12358e-10 ,3.68065e-10 ,5.41876e-11],

[-0.00012369 ,9.31717e-06 ,1.99106e-05 ,1.37436e-05 ,1.343e-05 ,1.45763e-05 ,1.40244e-05 ,1.17707e-05 ,1.04726e-05 ,8.71363e-06 ,5.71978e-06 ,6.10332e-06 ,4.19624e-06 ,3.32501e-06 ,2.14285e-06 ,2.07199e-06 ,1.41683e-06 ,1.00748e-06 ,1.12671e-06 ,3.89253e-07 ,4.26123e-07 ,1.17706e-07 ,4.75352e-08 ,1.3717e-08 ,2.93556e-09 ,8.61232e-10 ,3.18911e-10 ,2.85108e-10 ,4.22529e-11],

[-0.000114365 ,8.76424e-06 ,1.79284e-05 ,1.05431e-05 ,1.13264e-05 ,1.17234e-05 ,1.17707e-05 ,9.89392e-06 ,9.0174e-06 ,7.25303e-06 ,5.24238e-06 ,4.19624e-06 ,4.55104e-06 ,2.48968e-06 ,1.75827e-06 ,1.83802e-06 ,1.18058e-06 ,8.91197e-07 ,8.96849e-07 ,3.88655e-07 ,3.38622e-07 ,9.3797e-08 ,3.78376e-08 ,1.08772e-08 ,2.30653e-09 ,6.70937e-10 ,2.49109e-10 ,2.21233e-10 ,3.21301e-11],

[-7.12598e-05 ,1.39715e-05 ,1.7118e-05 ,1.00345e-05 ,1.02427e-05 ,9.28979e-06 ,9.62616e-06 ,7.91889e-06 ,6.44473e-06 ,5.66281e-06 ,4.42283e-06 ,3.32501e-06 ,2.48968e-06 ,2.82142e-06 ,1.16564e-06 ,1.32691e-06 ,1.00737e-06 ,5.88166e-07 ,7.3702e-07 ,2.7999e-07 ,2.70572e-07 ,8.27572e-08 ,3.40491e-08 ,1.02059e-08 ,2.14009e-09 ,6.13561e-10 ,2.18002e-10 ,1.99364e-10 ,3.04453e-11],

[-3.00274e-05 ,7.13919e-06 ,1.03818e-05 ,7.41505e-06 ,5.84693e-06 ,6.64585e-06 ,6.19094e-06 ,4.80733e-06 ,4.16883e-06 ,3.63137e-06 ,2.74234e-06 ,2.14285e-06 ,1.75827e-06 ,1.16564e-06 ,1.38481e-06 ,6.70704e-07 ,5.63112e-07 ,5.12248e-07 ,3.89705e-07 ,2.16792e-07 ,1.23884e-07 ,6.78121e-08 ,2.82161e-08 ,8.64471e-09 ,1.81226e-09 ,5.15783e-10 ,1.79387e-10 ,1.67621e-10 ,2.64458e-11],

[-4.3572e-05 ,4.97064e-06 ,8.73318e-06 ,5.35857e-06 ,5.60431e-06 ,5.92297e-06 ,5.66206e-06 ,4.64559e-06 ,4.31131e-06 ,3.39338e-06 ,2.49661e-06 ,2.07199e-06 ,1.83802e-06 ,1.32691e-06 ,6.70704e-07 ,1.12128e-06 ,4.25501e-07 ,4.12389e-07 ,4.89473e-07 ,1.12276e-07 ,1.89907e-07 ,5.14933e-08 ,2.14674e-08 ,6.61182e-09 ,1.39145e-09 ,3.98027e-10 ,1.37901e-10 ,1.29424e-10 ,2.06755e-11],

[-3.34794e-05 ,5.44068e-06 ,6.8597e-06 ,4.08441e-06 ,4.65295e-06 ,3.90338e-06 ,3.92355e-06 ,3.57936e-06 ,2.81118e-06 ,2.41228e-06 ,1.71911e-06 ,1.41683e-06 ,1.18058e-06 ,1.00737e-06 ,5.63112e-07 ,4.25501e-07 ,6.53884e-07 ,1.81168e-07 ,2.66917e-07 ,1.91521e-07 ,6.86769e-08 ,4.18778e-08 ,1.77798e-08 ,5.65295e-09 ,1.18578e-09 ,3.35185e-10 ,1.12571e-10 ,1.09001e-10 ,1.81444e-11],

[-1.43785e-05 ,3.12404e-06 ,4.82149e-06 ,3.88158e-06 ,2.74512e-06 ,3.04697e-06 ,3.03936e-06 ,2.42184e-06 ,2.0481e-06 ,1.59998e-06 ,1.32059e-06 ,1.00748e-06 ,8.91197e-07 ,5.88166e-07 ,5.12248e-07 ,4.12389e-07 ,1.81168e-07 ,4.07884e-07 ,1.35871e-07 ,6.74259e-08 ,1.33446e-07 ,3.15383e-08 ,1.35303e-08 ,4.38496e-09 ,9.21157e-10 ,2.59916e-10 ,8.57912e-11 ,8.45812e-11 ,1.44649e-11],

[-3.26909e-05 ,3.75157e-06 ,5.26283e-06 ,2.62957e-06 ,2.75205e-06 ,2.98113e-06 ,3.07196e-06 ,2.33168e-06 ,2.14333e-06 ,1.86108e-06 ,1.27829e-06 ,1.12671e-06 ,8.96849e-07 ,7.3702e-07 ,3.89705e-07 ,4.89473e-07 ,2.66917e-07 ,1.35871e-07 ,3.77008e-07 ,3.26428e-08 ,6.80689e-08 ,2.31084e-08 ,1.00569e-08 ,3.35076e-09 ,7.02679e-10 ,1.97225e-10 ,6.3451e-11 ,6.39888e-11 ,1.13041e-11],

[-1.01833e-05 ,2.68886e-06 ,2.1547e-06 ,1.30385e-06 ,1.64029e-06 ,1.1969e-06 ,1.09983e-06 ,1.06134e-06 ,8.63974e-07 ,7.30629e-07 ,5.83616e-07 ,3.89253e-07 ,3.88655e-07 ,2.7999e-07 ,2.16792e-07 ,1.12276e-07 ,1.91521e-07 ,6.74259e-08 ,3.26428e-08 ,1.39746e-07 ,-1.56018e-08 ,1.68742e-08 ,7.35405e-09 ,2.44425e-09 ,5.14499e-10 ,1.44619e-10 ,4.66733e-11 ,4.72601e-11 ,8.3611e-12],

[-4.34014e-06 ,1.24177e-06 ,1.68715e-06 ,1.51784e-06 ,1.09939e-06 ,1.09462e-06 ,1.05421e-06 ,9.1409e-07 ,7.63118e-07 ,5.66107e-07 ,4.41204e-07 ,4.26123e-07 ,3.38622e-07 ,2.70572e-07 ,1.23884e-07 ,1.89907e-07 ,6.86769e-08 ,1.33446e-07 ,6.80689e-08 ,-1.56018e-08 ,1.15224e-07 ,1.42613e-08 ,6.58765e-09 ,2.40731e-09 ,5.0775e-10 ,1.40326e-10 ,4.12883e-11 ,4.57564e-11 ,9.02947e-12],

[2.49897e-07 ,3.85689e-07 ,5.26071e-07 ,5.23056e-07 ,4.56338e-07 ,3.95721e-07 ,3.4493e-07 ,2.90174e-07 ,2.38877e-07 ,2.03687e-07 ,1.50204e-07 ,1.17706e-07 ,9.3797e-08 ,8.27572e-08 ,6.78121e-08 ,5.14933e-08 ,4.18778e-08 ,3.15383e-08 ,2.31084e-08 ,1.68742e-08 ,1.42613e-08 ,2.68513e-08 ,-6.16815e-09 ,2.52878e-09 ,-9.87321e-10 ,2.73453e-10 ,-1.45032e-10 ,6.55931e-11 ,-3.60592e-11],

[3.73215e-10 ,1.0197e-07 ,1.96708e-07 ,2.0845e-07 ,1.84685e-07 ,1.60253e-07 ,1.40514e-07 ,1.1824e-07 ,9.68253e-08 ,8.32451e-08 ,6.03985e-08 ,4.75352e-08 ,3.78376e-08 ,3.40491e-08 ,2.82161e-08 ,2.14674e-08 ,1.77798e-08 ,1.35303e-08 ,1.00569e-08 ,7.35405e-09 ,6.58765e-09 ,-6.16815e-09 ,5.01316e-09 ,-1.62286e-09 ,9.69918e-10 ,-1.55131e-10 ,1.28053e-10 ,-4.63957e-11 ,3.0714e-11],

[-6.92071e-08 ,-8.67471e-09 ,4.48683e-08 ,5.72827e-08 ,5.29021e-08 ,4.61024e-08 ,4.11047e-08 ,3.47436e-08 ,2.80824e-08 ,2.46654e-08 ,1.72088e-08 ,1.3717e-08 ,1.08772e-08 ,1.02059e-08 ,8.64471e-09 ,6.61182e-09 ,5.65295e-09 ,4.38496e-09 ,3.35076e-09 ,2.44425e-09 ,2.40731e-09 ,2.52878e-09 ,-1.62286e-09 ,1.53454e-09 ,-1.53987e-10 ,2.60667e-10 ,-5.02936e-11 ,4.69131e-11 ,-1.55509e-11],

[-1.15952e-08 ,4.23436e-10 ,1.11743e-08 ,1.34373e-08 ,1.22299e-08 ,1.05885e-08 ,9.23851e-09 ,7.68422e-09 ,6.1508e-09 ,5.28874e-09 ,3.71628e-09 ,2.93556e-09 ,2.30653e-09 ,2.14009e-09 ,1.81226e-09 ,1.39145e-09 ,1.18578e-09 ,9.21157e-10 ,7.02679e-10 ,5.14499e-10 ,5.0775e-10 ,-9.87321e-10 ,9.69918e-10 ,-1.53987e-10 ,3.77497e-10 ,-2.09171e-11 ,4.90332e-11 ,-1.22038e-11 ,1.12188e-11],

[-1.63972e-09 ,1.22621e-09 ,3.80469e-09 ,4.23108e-09 ,3.78053e-09 ,3.25238e-09 ,2.78634e-09 ,2.29329e-09 ,1.82504e-09 ,1.54136e-09 ,1.09564e-09 ,8.61232e-10 ,6.70937e-10 ,6.13561e-10 ,5.15783e-10 ,3.98027e-10 ,3.35185e-10 ,2.59916e-10 ,1.97225e-10 ,1.44619e-10 ,1.40326e-10 ,2.73453e-10 ,-1.55131e-10 ,2.60667e-10 ,-2.09171e-11 ,1.06806e-10 ,-2.58253e-11 ,2.21409e-11 ,-7.68228e-12],

[1.02472e-09 ,1.40207e-09 ,1.76038e-09 ,1.69893e-09 ,1.459e-09 ,1.24513e-09 ,1.04283e-09 ,8.48471e-10 ,6.80386e-10 ,5.58364e-10 ,4.12358e-10 ,3.18911e-10 ,2.49109e-10 ,2.18002e-10 ,1.79387e-10 ,1.37901e-10 ,1.12571e-10 ,8.57912e-11 ,6.3451e-11 ,4.66733e-11 ,4.12883e-11 ,-1.45032e-10 ,1.28053e-10 ,-5.02936e-11 ,4.90332e-11 ,-2.58253e-11 ,3.02399e-11 ,-1.6115e-11 ,8.88968e-12],

[-4.93594e-11 ,7.60943e-10 ,1.47941e-09 ,1.55271e-09 ,1.35956e-09 ,1.15929e-09 ,9.71687e-10 ,7.84871e-10 ,6.20921e-10 ,5.11259e-10 ,3.68065e-10 ,2.85108e-10 ,2.21233e-10 ,1.99364e-10 ,1.67621e-10 ,1.29424e-10 ,1.09001e-10 ,8.45812e-11 ,6.39888e-11 ,4.72601e-11 ,4.57564e-11 ,6.55931e-11 ,-4.63957e-11 ,4.69131e-11 ,-1.22038e-11 ,2.21409e-11 ,-1.6115e-11 ,1.4775e-11 ,-6.53667e-12],

[-2.63688e-10 ,-1.78143e-11 ,1.95731e-10 ,2.4066e-10 ,2.18137e-10 ,1.85831e-10 ,1.55553e-10 ,1.24214e-10 ,9.55121e-11 ,7.89289e-11 ,5.41876e-11 ,4.22529e-11 ,3.21301e-11 ,3.04453e-11 ,2.64458e-11 ,2.06755e-11 ,1.81444e-11 ,1.44649e-11 ,1.13041e-11 ,8.3611e-12 ,9.02947e-12 ,-3.60592e-11 ,3.0714e-11 ,-1.55509e-11 ,1.12188e-11 ,-7.68228e-12 ,8.88968e-12 ,-6.53667e-12 ,5.67405e-12]]
    
    def __init__(self, data, cov, numCols, ifSimp=False):
        strArr=Util.strToStrArr(data, " ", numCols)
        self.E=np.array([float(x) for x in strArr[0]])
        self.S=np.array([float(x) for x in strArr[1]])
        self.delS=np.array(float(x[:-1])/100 for x in strArr[2])
        self.cov=cov
        self.E_avg=np.zeros(len(self.E)-1)
        self.R=np.zeros(len(self.E)-1)
        self.binEdge=dayaBay250.bins
        self.calcR()
        self.delR=np.zeros(len(self.R))
        if not ifSimp:
            for x in range(len(self.R)):
                self.delR[x]=math.sqrt(self.S[x+1]**(-2)*cov[x][x]+self.S[x]**2*self.S[x+1]**(-4)*cov[x+1][x+1]-2*self.S[x]*self.S[x+1]**(-3)*cov[x][x+1])
        
    @staticmethod
    def dayaSpectra():
        """Creates a Spectrum (Daya Bay spectrum specifically) from collected Daya Bay data"""
        return dayaBay250(dayaBay250.dayaData, dayaBay250.dayaCov, 3)
        
@staticmethod
def getReactDatabaseV2(percent, endFile):
        """Creates reactor nuclide database for the specific reactor fission fractions.

        Args:
            percent (dictionary of 4 floats): the reactor fission fractions for u5, u8, pu9, pu1 (using those as dictionary keys)
            endFile (string): place to save final file to
            bins (arraylike of floats, optional): the bin edges used. Defaults to np.arange(0, 10.01, .01).

        Returns:
            Sum Spectra: CFY for the specific nucl
        """  
        dictWeight={}
        for fuel in ["u5", 'u8', 'pu9', 'pu1']:
            fileName="Nuclide Spectral Data\\"+fuel+"cfy.txt"
            with open(fileName, "r") as f:
                data=f.read()
            strArr=Util.strToStrArr(data, delimeter=';', numCols=4)
            arr=[float(strArr[3][i]) for i in range(len(strArr[3]))]
            print(np.sum(arr))
            
            for i in range(len(strArr[0])):
                strKey="z_"+str(strArr[0][i])+"_a_"+str(strArr[1][i])+"_liso_"+str(strArr[2][i])
                if strKey not in nuclideSpectra.names:
                    strKey+="_t"
                if strKey not in dictWeight:
                    dictWeight[strKey]=float(strArr[3][i])*percent[fuel]
                else:
                    dictWeight[strKey]+=float(strArr[3][i])*percent[fuel]
        d=Util.readDict(nuclideSpectra, "rawNuclideSpectra.csv")
        integ=0
        for x in dictWeight:
            integ+=dictWeight[x]
        print(integ)
        for x in d:
            d[x].S=np.array(d[x].S[:len(sumSpectra.crossSection)])*dictWeight[x]*sumSpectra.crossSection[:len(d[x].S)]
            d[x].E=d[x].E[:len(d[x].S)]
        Util.dictToCSV(endFile, d)
    