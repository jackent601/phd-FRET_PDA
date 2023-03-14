# Python Modules
import pandas as pd
import numpy as np
from scipy.optimize import minimize
import sys
import os
import pickle

# PDA modules
import PDA as PDA

# FRET modules
sys.path.append('./../')
import FRET_modules.modulesCorrectionFactorsAndPlots as MCF
import matplotlib.pyplot as plt

class kineticFitting():
    """
    burstData: pandas dataframe of burst data
                    -> Must be corrected, i.e. have E3/S3_avg_beta_gamma_columns
                    
    EBins: Array, E - bins used in comparing experimental to simulated 
    EBinCentres
    metaData: Dictionary, contains information on object
    
    S = 'S3_avg_beta_gamma', assumes dataframe has been corrected
    CDEMasks: Filters placed on static/dynamic populations, defaults 20 each, less important in this analysis

    PDA Parameters
    defaults
    N: 50
    r0: 50
    rDev: 5
    gaussian resolution: 20
    """
    S_featureName = 'S3_avg_beta_gamma'
    E_featureName = 'E3_avg_beta_gamma'
    CDEMasks = [20, 20]
    staticPopulationELimits = [[0,0.5],[0.501, 1]]
    
    def __init__(self, 
                 rawBurstData, 
                 EBins, 
                 EBinCentres, 
                 metaData, 
                 N=50, 
                 r0=50, 
                 rDeviation=5, 
                 gaussianResolution=20):
        # MetaData
        self.metaData = metaData
        
        # Experimental Burst Data
        self.rawBurstData = rawBurstData
        
        # Doner-Only, Acceptor-Only populations removed
        ADBurstData = MCF.typical_S_ALEX_FRET_CDE_filter(self.rawBurstData,
                                                         S = self.rawBurstData[self.S_featureName],
                                                         FRET2CDEmask = 200000)
        
        # Then remove E outside of E-bins
        Emask = np.logical_and(ADBurstData[self.E_featureName]<=EBins[-1], ADBurstData[self.E_featureName]>=EBins[0])
        self.validBurstData = ADBurstData[Emask]
        
        # Dynamic population removed
        self.staticBurstData = MCF.typical_S_ALEX_FRET_CDE_filter(self.rawBurstData, 
                                                                  S = self.rawBurstData[self.S_featureName],
                                                                  ALEX2CDEmask = self.CDEMasks[0], 
                                                                  FRET2CDEmask = self.CDEMasks[1])        
        
        # E-Bins used to compare histogram
        self.EBins = EBins
        self.EBinCentres = EBinCentres
        self.dBin = EBinCentres[1]-EBinCentres[0]
        
        # Static and Dynamic E-Histogram
        self.experimentalEHist = self.getStaticAndDynamicExperimentalHistogram()
        
        # Static E-Histogram
        self.staticExperimentalEHist = self.getStaticExperimentalHistogram()
        
        # Static E1, E2 values
        self.static_E1, self.static_E2 = self.getStaticE1E2values()
        
        # PDA Parameters
        self.N=N
        self.r0=r0 
        self.rDeviation=rDeviation 
        self.gaussianResolution=gaussianResolution
        
        
        # Kinetic Results
        self.kineticResults=[]
        self.optimisationResults=[]

        
    # Exerpimental E histogram
    def getStaticAndDynamicExperimentalHistogram(self):
        """Uses data with both static and dynamic populations"""
        staticAndDynamicEHist, _ = np.histogram(self.validBurstData[self.E_featureName], self.EBins)
        return staticAndDynamicEHist
    
    # Exerpimental E histogram
    def getStaticExperimentalHistogram(self):
        """Uses data with dynamic populations removed"""
        staticEHist, _ = np.histogram(self.staticBurstData[self.E_featureName], self.EBins)
        return staticEHist
    
    # E1, E2
    def getStaticE1E2values(self):
        b, g, expEs, expSs = MCF.calculate_beta_and_gamma_correction(None,
                                                                     population_E_limits = self.staticPopulationELimits,
                                                                     E = self.staticBurstData[self.E_featureName],
                                                                     S = self.staticBurstData[self.S_featureName],
                                                                     sample_percent = 0.05,
                                                                     plot_fig = False)
        static_E1, static_E2 = expEs
        return static_E1, static_E2
    
    # Plotting
    def plotEHist(self, title=None):
        plt.bar(self.EBinCentres, 
                self.experimentalEHist, 
                width=self.dBin, 
                alpha=0.5)
        plt.ylabel('count')
        plt.xlabel('E')
        _title = 'E-Hist' if title is None else title
        plt.title(_title)
        plt.show()
        
    def plotStaticEHist(self, title=None):
        plt.bar(self.EBinCentres, 
                self.staticExperimentalEHist, 
                width=self.dBin, 
                alpha=0.5)
        plt.ylabel('count')
        plt.xlabel('E')
        _title = 'Static E-Hist' if title is None else title
        plt.title(_title)
        plt.show()
    
    def plotStaticAndDynamicEHist_Normalised(self, title=None):
        plt.bar(self.EBinCentres, 
                self.experimentalEHist/sum(self.experimentalEHist),
                width=self.dBin, 
                alpha=0.5, 
                label='E-Hist')
        plt.bar(self.EBinCentres, 
                self.staticExperimentalEHist/sum(self.staticExperimentalEHist), 
                width=self.dBin, 
                alpha=0.5,
                label='Static E-Hist')
        plt.legend()
        plt.ylabel('count (Normalised)')
        plt.xlabel('E')
        _title = 'E-Hist Comparison (Normalised)' if title is None else title
        plt.title(_title)
        plt.show()
        
    # PDA
    def get_pEsnWithGauss(self, nBurstBins, k1, kminus1, E1=None, E2=None, N=None, r0=None, rDeviation=None, gaussianResolution=None, plot=True):
        """
        Bins burst durations first to speed up optimisation, option for PDA parameters to be overwritten
        Saves results to a results list tied to object
        """
        # Unpack defaults if necessary
        E1 = E1 if E1 is not None else self.static_E1
        E2 = E2 if E2 is not None else self.static_E2
        N = N if N is not None else self.N
        r0 = r0 if r0 is not None else self.r0
        rDeviation = rDeviation if rDeviation is not None else self.rDeviation
        gaussianResolution = gaussianResolution if gaussianResolution is not None else self.gaussianResolution
        
        # Run kinetic simulation
        pEsn = PDA.fastFindpEsn_fromBurstDataFrame_withGaussian(BurstData=self.validBurstData, 
                                                                nBurstBins=nBurstBins, 
                                                                Ebins=self.EBins, 
                                                                EbinCentres=self.EBinCentres,
                                                                E1=E1,
                                                                E2=E2,
                                                                N=N,
                                                                k1=k1,
                                                                kminus1=kminus1,
                                                                r0=r0,
                                                                rDeviation=rDeviation,
                                                                gaussianResolution=gaussianResolution)
        # Get SSE
        sse = self.getSSEpEsn(pEsn)
        
        # Save results in object
        meta = {'method': 'BinnedGaussian', 'nBurstBins': nBurstBins, 'r0': r0, 'rDeviation': rDeviation, 'gaussianResolution': gaussianResolution}
        _res = {'k1': k1, 'kminus1': kminus1, 'pEsn': pEsn, 'SSE': sse, 'E1': E1, 'E2': E2, 'metaData': meta}
        self.addKineticResult(_res)
        return pEsn
    
    def plot_pEsn(self, pEsn, title=None, savePath=None):
        # Plot Experimental
        plt.bar(self.EBinCentres, 
                self.experimentalEHist,
                width=self.dBin, 
                alpha=0.5, 
                label='Exp E-Hist')
        # Plot Simulated
        plt.bar(self.EBinCentres, 
                pEsn, 
                width=self.dBin, 
                alpha=0.5,
                label='Sim E-Hist')
        plt.legend()
        plt.ylabel('Count')
        plt.xlabel('E')
        _title = 'E-Hist Comparison (Exp vs Sim)' if title is None else title
        plt.title(_title)
        if savePath is not None:
            plt.savefig(savePath)
            plt.clf()
        else:
            plt.show()
        
    def getSSEpEsn(self, pEsn):
        return np.sum(np.array(np.array(pEsn) - np.array(self.experimentalEHist))**2)
    
    def addKineticResult(self, result):
        self.kineticResults.append(result)
        
    def addKineticResults(self, results):
        for result in results:
            self.addKineticResult(result)    
        
    def getKineticResults(self):
        return self.kineticResults
    
    def addOptimisationResult(self, result):
        self.optimisationResults.append(result)
        
    def addOptimisationResults(self, results):
        for result in results:
            self.addOptimisationResult(result)    
        
    def getOptimisationResults(self):
        return self.optimisationResults
    
    def optimiseBinnedPDALossFunction(self, 
                                      x, 
                                      nBurstBins, 
                                      E1=None, 
                                      E2=None, 
                                      N=None, 
                                      r0=None, 
                                      rDeviation=None, 
                                      gaussianResolution=None):
        """Very similar to get_pEsnWithGauss but k's are array to be optimised"""
        _pEsn = self.get_pEsnWithGauss(nBurstBins=nBurstBins, k1=x[0], kminus1=x[1], E1=E1, E2=E2, N=N, r0=r0, rDeviation=rDeviation, gaussianResolution=gaussianResolution)
        sse = self.getSSEpEsn(_pEsn)
        return sse
    
    def optimiseBinnedPDA(self, 
                          saveDir,
                          intial_ks,
                          nBurstBins, 
                          optimiseMethod = 'L-BFGS-B',
                          optimiseOptions = {'eps': [1, 1], 'ftol': 1e-10, 'gtol': 1e-10, 'maxiter':50},
                          kbounds=((5,5000), (5,5000))):
        """
        Each itertion will save result to list stored in self
        callback function can also plot the results for convenient debugging by fetching last stored result in this list
        """
        # Create directory for figures
        figDir = os.path.join(saveDir, 'figs')
        os.makedirs(figDir, exist_ok=True)

        # Call back debug
        _callback=lambda kpair: self.plot_pEsn(pEsn=self.getKineticResults()[-1]['pEsn'], 
                                               title=f'Optimising k1: {self.getKineticResults()[-1]["k1"]:0.2f}, kminus1: {self.getKineticResults()[-1]["kminus1"]:0.2f}, SSE: {self.getKineticResults()[-1]["SSE"]:0.0f}', 
                                               savePath=os.path.join(figDir, f'Opt_{len(self.getKineticResults())}.png'))
        # _callback=lambda kpair: print(f'Ran k1: {kpair[0]:0.2f}, k-1: {kpair[1]:0.2f}')
        
        # Currently unbounded with not optimisation options
        optResult = minimize(self.optimiseBinnedPDALossFunction,
                             args=(nBurstBins),
                             x0=np.array([intial_ks[0], intial_ks[1]]),
                             bounds=kbounds, 
                             callback=_callback,
                             method=optimiseMethod,
                             options=optimiseOptions)
        
        # Add result
        _optResultEntry = {'optResult': optResult, 'initial_ks': intial_ks, 'method': optimiseMethod, 'options': optimiseOptions, 'nBurstBins': nBurstBins}
        self.addOptimisationResult(_optResultEntry)
        
        # Plot results
        k1_Opt, kminus1_Opt = optResult.x
        pEsn_Opt = self.get_pEsnWithGauss(nBurstBins, k1_Opt, kminus1_Opt)
        self.plot_pEsn(pEsn_Opt, title = f'Optimised, k1: {k1_Opt:0.2f}, kminus1: {kminus1_Opt:0.2f}', savePath=os.path.join(saveDir, "optimised.png"))
        print(f'Burst Count: {sum(self.experimentalEHist)}, ({len(self.validBurstData)}) Simulated Count: {sum(pEsn_Opt):0.0f}')
        print(optResult)
    
    def __str__(self):
        s = "E-Acquisition & Kinetic Fitting Object\n"
        s += f"Meta-Info\n"
        for key, val in self.metaData.items():
            s += f"\t {key}: {val}\n"
        s += f"\t static E1: {self.static_E1:0.2f}\n\t static E2: {self.static_E2:0.2f}\n"
        return s