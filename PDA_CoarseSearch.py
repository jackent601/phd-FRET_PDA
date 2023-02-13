from scipy.special import i0, i1
from scipy.stats import binom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import TwoStateKineticModel as KM2S

# ================================================================================================================================
#   PROBABILITY DISTRIBUTION ANALYSIS
# ================================================================================================================================

# Method described here: https://pubmed.ncbi.nlm.nih.gov/20575136/

# Takes data from FRET experiments and determines values for E1, E2, k1, kminus1 
# assuming a two state kinetic model (TwoStateKineticModel.py in directory)

# Includes method for speeding up 'coarse search' by:
#     Fixing E1, E2 guesses by using FRET2CDE filtering to find static FRET bursts
#     First bining burst durations to avoid multiple CFD generation and scaling the sampling of each CFD generated

# ================================================================================================================================


# ================================================================================================================================
# HIGH RESOLUTION SEARCHING (BASED ON INDIVIDUAL BURST DURATIONS)
# ================================================================================================================================
def getDurationsAndFlourescence(sample, durationFeatureName='duration', adPhotonsFeatureName='AD',ddPhotonsFeatureName='DD'):
    """
    Each burst has a duration (s) used to generte T distribution and flourescence (#photons) used for binomial sampling
    Returns numpy arrays of durations and flourescences
    """
    durations = sample[durationFeatureName]
    flourescenceRaw = sample[adPhotonsFeatureName] + sample[ddPhotonsFeatureName]
    flourescence = [int(F) for F in flourescenceRaw]
    
    return np.array(durations), np.array(flourescence)

def getEsnfEsnFromDurationsAndFs(E1, E2, durations, Fs, K, N, k1, kminus1):
    """
    Calculates Esnf, Esn for a list of durations with flourescences
    """

    # Get arrays of T1, and T2 from CFD distributions
    allT1s, allT2s = KM2S.p_getT1T2SamplesFromMultipleBurstDurations(durations, K, {'N': N, 'k_1': k1, 'k_minus1': kminus1})
    
    # Scale Flourescence by oversampling factor for binomial sampling
    F_scaled = Fs.repeat(K)

    # Calculate shot-noise free simulated E values 
    E_snf = KM2S.E_snf_TwoStatesEqualBrightness(allT1s, E1, allT2s, E2)

    # Calculate shot-noise-dependent for each T pair in this duration sample
    E_sn = np.array([KM2S.getEShotNoise(_F, _E_snf) for _F, _E_snf in zip(F_scaled, E_snf)]).ravel() 
    
    return E_snf, E_sn

def getEsnfEsnFromBurstDataFrame(BurstData, E1, E2, K, N, k1, kminus1):
    d, F = getDurationsAndFlourescence(BurstData)
    return getEsnfEsnFromDurationsAndFs(E1=E1, E2=E2, durations=d, Fs=F, K=K, N=N, k1=k1, kminus1=kminus1)

# ================================================================================================================================
# COARSE SEARCHING BASED ON BINNING BURST DURATIONS
# ================================================================================================================================
# def getMultipleT1T2SamplesFromBurstDurationHistogram(overSampleFactorK, 
#                                                      burstDurationBinValues, 
#                                                      burstDurationBinCounts, 
#                                                      PTwoStateParameters, 
#                                                      seed=None):
#     """
#     Fast method for generating a simulated T1, T2s
#     Takes binned burst duration values and generates CFD only for bin values rather than every burst
#     Scales sampling of each CFD based on counts within each bin
#     returns dictionary of with keys being burstDurations
#     """
#     if seed is not None:
#         np.random.seed(seed)

#     allPairs = {}
#     for burstDuration, burstCount in zip(burstDurationBinValues, burstDurationBinCounts):
#         scaledSampleFactor = overSampleFactorK*burstCount
#         TPairs = KM2S.getMultipleT1T2SamplesFromBurstDuration(scaledSampleFactor, burstDuration, PTwoStateParameters, seed=None)
#         allPairs[burstDuration] = {'TPairs': TPairs, 'binCount':burstCount}
#     return allPairs

# def getF_andT_valuesForDurationBins(burstDataFrame, durationBins, binCentres, durationHist):
#     """
#     For each time bin, gathers a list of:
#         all F values (total flourescence)
#             F values used in binomial sampling of E shot-noise
#         all duration values

#     F values required for binomial sampling
    
#     returns dictionary of dictionaries with {E_bin_Centre1: {FVals [F1, F2, ... Fx], Durations: [t1, t2, ... tx], ...}
#     durationHist provides a sanity check that there are no mismatch counting
#     """
#     burstDurations = burstDataFrame['duration']
    
#     burstBinFValues = {}
#     for i, TBin in enumerate(durationBins[:-1]):    
#         binCentre = binCentres[i]
        
#         # Filter for time bin
#         TBinMask = np.logical_and((burstDurations >= TBin), (burstDurations < durationBins[i+1]))
#         _TBinDF = burstDataFrame[TBinMask]

#         # Total Flourescence of burst ('flourescence considering FRET events only, hence AD and DD, not AA?)
#         BSs = list(_TBinDF.DD + _TBinDF.AD)

#         # Take integer (float due to background subtraction)
#         BSs = [int(bs) for bs in BSs]
        
#         # Get T values
#         durationVals = list(burstDurations.values[TBinMask])
        
#         # Sanity Checks
#         assert len(durationVals) == durationHist[i], "Error, count of points within bin does not match duration count!"
#         assert len(BSs) == durationHist[i], "Error, count of points within bin does not match BS count!"
        
#         burstBinFValues[binCentre] = {'Count': durationHist[i], 'F_values': BSs, 'durationValues': durationVals}
        
#     return burstBinFValues

# def getEvaluesFromDictionaryOfTPairs(E1, E2, overSampleFactorK, durationBinCentres, simulatedTPairsDict, FvaluesBinDict):
#     """
#     Importantly requires a dictionary (with the same keys!) of F values for each duration bin, F values are
#     what are used to sample from a binomial distribution of acceptor photons
#     Note, doesnt matter which E value is paired to each F value as all E values drawn from the same distribution (i.e. all F have same duration)

#     Returns dictionary keys are bin values, values are E dictionary
#     """
#     EvalueDict = {}

#     for binCentre in durationBinCentres:
#         # Get TPairs and F values for this duration
#         TPairs = simulatedTPairsDict[binCentre]['TPairs']
#         FValues =  FvaluesBinDict[binCentre]['F_values']

#         # Calculate shot-noise free simulated E values
#         E_snf = getEsnfForListOfRepeatSimulatedPairs(E1, E2, TPairs)

#         # Check sampling dimensions correct otherwise E sampling will fail
#         assert overSampleFactorK*len(FValues) == len(TPairs), "k*Ncount does not equal number of E values, sampling dimensions wrong"

#         # Calcualte the E shot-noise, with appropraite sampling
#         # Note, doesnt matter which E value is paired to each F value as all E values drawn from the same distribution
#         E_sn = []
#         for i, _F in enumerate(FValues):
#             for _E_snf in E_snf[i*overSampleFactorK:((i+1)*overSampleFactorK)]:
#                     # Uses fast method which uses optimised numpy binomial avoids generating random number separtely
#                     E_sn.append(KM2S.getEShotNoise(_F, _E_snf))

#         # Add to Dictionary
#         EvalueDict[binCentre] = {'Esnf':E_snf, 'Esn':E_sn}
#     return EvalueDict


# def getFlatEvaluesFromDictionaryOfTPairs(E1, 
#                                          E2,
#                                          overSampleFactorK,
#                                          durationBinCentres,
#                                          simulatedTPairsDict,
#                                          FvaluesBinDict):
#     """
#     Very similar to getEvaluesFromDictionaryOfTPairs but flattens the returned dictionary ready for
#     histogram analysis of simulated Es
#     """
#     EvalueDict = getEvaluesFromDictionaryOfTPairs(E1,
#                                                       E2,
#                                                       overSampleFactorK,
#                                                       durationBinCentres,
#                                                       simulatedTPairsDict,
#                                                       FvaluesBinDict)

#     # Flatten Esnf, and Esn
#     sim_Esnf = [e for k, d in EvalueDict.items() for e in d['Esnf']]
#     sim_Esn = [e for k, d in EvalueDict.items() for e in d['Esn']]
    
#     return sim_Esnf, sim_Esn











# """
# SIMULATED E UTILITIES
# """
# def getEsnfForListOfRepeatSimulatedPairs(E1, E2, TSimulatedPairs):
#     """
#     Expects an input of [ [T1, T2], [T1, T2],  ...]
#     """
#     E_simulated = []
#     for Pair in TSimulatedPairs:
#         # Unpack
#         _T1, _T2 = Pair
#         # Calculate
#         E_simulated.append(KM2S.E_snf_TwoStatesEqualBrightness(T1=_T1, E1=E1, T2=_T2, E2=E2))
    # return E_simulated

def getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK):
    simulated_E_Hist, _ = np.histogram(simulatedEs, Ebins)
    simulated_E_Hist_Scaled = simulated_E_Hist/overSamplingFactorK
    return simulated_E_Hist_Scaled


# # # # # # # # # # BREAK # # # # # # # # # #

def getSSEAndHistFromListOfEs(simulatedEs, Ebins, overSamplingFactorK, experimentEHist):
    # Get Hist
    simulated_E_Hist_Scaled = getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK)
    # Return SSE
    return np.sum(np.array(np.array(experimentEHist) - np.array(simulated_E_Hist_Scaled))**2), simulated_E_Hist_Scaled

def getSSEOnlyFromListOfEs(simulatedEs, Ebins, overSamplingFactorK, experimentEHist):
    """
    Quicker as doesnt return hist
    """
    # Get Hist
    simulated_E_Hist_Scaled = getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK)
    # Return SSE
    return np.sum(np.array(np.array(experimentEHist) - np.array(simulated_E_Hist_Scaled))**2)

# """
# END TO END E HIST COMPARISONS - SANITY CHECKS MAINLY
# """
# def endToEndEHistCompare(experimentalObjectsDict,
#                       PTwoStateParameters,
#                       E1,
#                       E2,
#                       ShotNoiseFree = False,
#                       ShotNoise = True,
#                       plothists = True):
#     # unpack experimental objects
#     overSampleFactorK = experimentalObjectsDict['overSampleFactorK']
#     burstDurationBinValues = experimentalObjectsDict['burstDurationBinValues']
#     burstDurationBinCounts = experimentalObjectsDict['burstDurationBinCounts']
#     seed = experimentalObjectsDict['seed']
#     FvaluesBinDict = experimentalObjectsDict['FvaluesBinDict']
#     Ebins = experimentalObjectsDict['Ebins']
#     EBinCentres = experimentalObjectsDict['EBinCentres']
#     ExperimentEHist = experimentalObjectsDict['ExperimentEHist']
    
#     # For Each bin simulate a range of T pairs (Soon for a range of ks)
#     simulatedTPairsDict = getMultipleT1T2SamplesFromBurstDurationHistogram(overSampleFactorK,
#                                                                                burstDurationBinValues,
#                                                                                burstDurationBinCounts,
#                                                                                PTwoStateParameters,
#                                                                                seed=seed)

#     # Calculate Esnf, and Esn for each bin
#     sim_Esnf, sim_Esn = getFlatEvaluesFromDictionaryOfTPairs(E1,
#                                                                  E2,
#                                                                  overSampleFactorK,
#                                                                  burstDurationBinValues,
#                                                                  simulatedTPairsDict,
#                                                                  FvaluesBinDict)
    
#     # Get Hists
#     k1 = PTwoStateParameters['k_1']
#     kminus1 = PTwoStateParameters['k_minus1']
#     dBin = Ebins[1]-Ebins[0]
#     if ShotNoise:
#         SSE_sn, scaledSimulatedEsnHist = getSSEAndHistFromListOfEs(sim_Esn, Ebins, overSampleFactorK, ExperimentEHist)
#         if plothists:
#             plt.bar(EBinCentres, ExperimentEHist, width=dBin, alpha =0.5, label='exp')
#             plt.bar(EBinCentres, scaledSimulatedEsnHist, width=dBin, alpha =0.5, label='Simulated Esn')
#             plt.title(f'Simulated E histogram shot-noise\nk1: {k1}, kminus1: {kminus1}, SSE: {int(SSE_sn)}')
#             plt.legend()
#             plt.show()
        
#     if ShotNoiseFree:
#         SSE_snf, scaledSimulatedEsnfHist = getSSEAndHistFromListOfEs(sim_Esnf, Ebins, overSampleFactorK, ExperimentEHist)
#         if plothists:
#             dBin = EBinCentres[1] - EBinCentres[0]
#             plt.bar(EBinCentres, ExperimentEHist, width=dBin, alpha =0.5, label='exp')
#             plt.bar(EBinCentres, scaledSimulatedEsnfHist, width=dBin, alpha =0.5, label='Simulated Esnf')
#             plt.title(f'Simulated E histogram shot-noise free\nk1: {k1}, kminus1: {kminus1}, SSE: {int(SSE_snf)}')
#             plt.legend()
#             plt.show()
  
# """
# DEPRECIATED
# """

# # def getEHistFromSimulatedTPairs(overSampleFactorK, TSimulatedPairs, E1, E2, EBins):
# #     """
# #     TSimulatedPairs should be a list of list of T pairs: [ [T1, T2], [T1, T2], ...]
# #     """
# #     # First Calculate Simulated Es
# #     _E_simulated = getEsnfForListOfRepeatSimulatedPairs(E1=E1, E2=E2, TSimulatedPairs=TSimulatedPairs)
    
# #     # Calculate simulated E Hist and Compare to Experiment
# #     EHistSim, _ = np.histogram(_E_simulated, bins = EBins)
# #     return EHistSim/overSampleFactorK

# # def getSSEFromSimulatedTPairs(overSampleFactorK, TSimulatedPairs, E1, E2, EBins, EHistExperiment):    
# #     """
# #     TSimulatedPairs should be a list of list of T pairs: [ [[T1, T2], [T1, T2], ...] , ...]
# #     """
# #     # Check Experimental Histogram same dimensions as generated histogram
# #     assert len(EHistExperiment) == len(EBins)-1, "Dimensions of Experimental Histogram and simulated histogram must match!"
    
# #     # Get Histogram For these simulated Ts
# #     EHistSim = getEHistFromSimulatedTPairs(overSampleFactorK=overSampleFactorK, TSimulatedPairs=TSimulatedPairs, E1=E1, E2=E2, EBins=EBins)
    
# #     # Return SSE
# #     return np.sum(np.array(np.array(EHistExperiment) - np.array(EHistSim))**2), EHistSim
















# """
# Utilitity

# Doesn't really belong here, could be anywhere, this is just to tidy up jupyter notebook
# """
# def getSamplePointsOverOrdersOfMag(O_StartPower, O_FinalPower, SampleNumber):
#     orders_of_mag = [i for i in range(O_StartPower, O_FinalPower+1)]
    
#     # Get all Sample Points
#     samplePoints = []
#     for O in orders_of_mag[:-1]:
#         samplePoints.append(np.linspace(10**O, 10**(O+1), SampleNumber))

#     # flatten all samples
#     return [point for points in samplePoints for point in points]