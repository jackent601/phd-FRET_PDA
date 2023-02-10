from scipy.special import i0, i1
from scipy.stats import binom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import TwoStateKineticModel as KM2S

# """
# T1, T2 PROBABILITY DISTRIBUTION FOR BURST DURATION

# Kinetic Model for a system that interconverts between two states E1, E2 with constants k1, kminus1

#     Equations ref: Detection of Structural Dynamics by FRET: A Photon Distribution and Fluorescence Lifetime Analysis of Systems with Multiple States
#     DOI: https://pubs.acs.org/doi/10.1021/jp102156t 

# i0, i1 are MODIFIED bessel functions, first kind, and of 0, 1 order respectively
# Note there is a difference between del_t (burst duration) and del_T (width of probability sample)
# """

# def P_TwoState_T1_0(k1, k2, del_t):
#     return (k1/(k1+k2))*np.exp(-k2*del_t)

# def P_TwoState_T2_0(k1, k2, del_t):
#     return P_TwoState_T1_0(k1=k2, k2=k1, del_t=del_t)

# def P_TwoState_T1(k1, k2, T1, T2, del_T):
#     bessel_arg = 2*np.sqrt(k1*k2*T1*T2)
    
#     coeff1 = (2*k1*k2)/(k1+k2)
#     bessel1 = i0(bessel_arg)
    
#     coeff2 = ((k2*T1+k1*T2)/(k1+k2))*((np.sqrt(k1*k2))/(np.sqrt(T1*T2)))
#     bessel2 = i1(bessel_arg)
    
#     exp_term = np.exp(-k1*T1-k2*T2)*del_T
    
#     return (coeff1*bessel1 + coeff2*bessel2)*exp_term

# def PTi_TwoState_at_del_t(N, burstDuration, k_1, k_minus1, cumsum = True, debug = False):
#     """
#     For given burst duration, k_1, k_minus1 calculates the probability distribution of spending t seconds in state 1
#     Returns a list of dictionaries of values
    
#     Equations ref: Detection of Structural Dynamics by FRET: A Photon Distribution and Fluorescence Lifetime Analysis of Systems with Multiple States
#     DOI: https://pubs.acs.org/doi/10.1021/jp102156t
    
#     Equation Derivation: Calculation of Photon-Count Number Distributions via Master Equations
    
#     Eqs in the limit of small Delta T, here Delta T = burstDuration/N
#     """
    
#     del_T = burstDuration/N
    
#     P_T1s = []
    
#     for n in range(N+1):

#         if n == 0:
#             # Primarily in State E2
#             _T1 = del_T/4
#             _T2 = burstDuration - _T1
#             _P_T1 = P_TwoState_T1_0(k1=k_1, k2=k_minus1, del_t=burstDuration) + P_TwoState_T1(k1=k_1, k2=k_minus1, T1=_T1, T2=_T2, del_T=del_T/2)
#         elif n == N:
#             # Primarily in State E1
#             _T2 = del_T/4
#             _T1 = burstDuration - _T2
#             _P_T1 = P_TwoState_T2_0(k1=k_1, k2=k_minus1, del_t=burstDuration) + P_TwoState_T1(k1=k_1, k2=k_minus1, T1=_T1, T2=_T2, del_T=del_T/2)
#         else:
#             # Transition Between States
#             _T1 = n*del_T
#             _T2 = burstDuration - _T1
#             _P_T1 = P_TwoState_T1(k1=k_1, k2=k_minus1, T1=_T1, T2=_T2, del_T=del_T)

#         P_T1s.append({'T1':_T1, 'T2':_T2, 'PT1': _P_T1, 'i': n})
                
#         if debug:
#             print(f'T1: {_T1*1e3:0.2f} ms, T2: {_T2*1e3:0.2f} ms, P_T1: {_P_T1}')
        
#     if np.any(np.array([d['PT1'] for d in P_T1s]) < 0):
#             print("WARNING NEGATIVE PROBABILITIES ENCOUNTERED")
        
#     P_T1s_df = pd.DataFrame(P_T1s)
    
#     if cumsum:
#         P_T1s_df['cumsum'] = P_T1s_df.PT1.cumsum()
            
#     return P_T1s_df

# def getT1T2SampleFromTiCFD(MC_sample, burstDuration, T_Vals, T_CFD):
#     """
#     MC_sample: monte carlo sample value
#     T_Vals: values for T1
#     T_CFD: T1 cumulative frequency distribution values

#     burstDuration must be value used to generate the CFD
#     """
#     # Catch if MC draw greater than max cumsum (this is due to discrete distribution generation)
#     if MC_sample >= max(T_CFD):
#         return T_Vals[-1], burstDuration-T_Vals[-1]
#     if MC_sample <= min(T_CFD):
#         return T_Vals[0], burstDuration-T_Vals[0]

#     # Get first location where probability exceeds sample
#     cumsumIndex = np.argwhere(T_CFD >= MC_sample)[0, 0]

#     # Retrieve corresponding T1 value, hence calculate T2 value
#     T1 = T_Vals[cumsumIndex]
#     T2 = burstDuration - T1
#     return T1, T2

# """
# E SNF (shot-noise free) AND E SN (shot-noise)
# """
    
# def E_snf_TwoStates(Q1, T1, E1, Q2, T2, E2):
#     """
#     Ref: Detection of Structural Dynamics by FRET: A Photon Distribution and Fluorescence Lifetime Analysis of Systems with Multiple States
#     """
#     return (Q1*T1*E1 + Q2*T2*E2)/(Q1*T1 + Q2*T2)

# def E_snf_TwoStatesEqualBrightness(T1, E1, T2, E2):
#     """
#     Assumes Equal Brightness Q1 = Q2, need a referance discussing appropriateness
#     Q, brightness is the total count rate (in both donor and acceptor detection channels) at a given excitiation intensity,
#     at a concentration of one moleucle per observation volume
#     """
#     return E_snf_TwoStates(Q1=1, T1=T1, E1=E1, Q2=1, T2=T2, E2=E2)

# def getEShotNoise(F, Esnf):
#     """
#     Uses numpy module to sample from binomial which is much faster, but loses tracability on MC draw
#     """
#     AD_draw = np.random.binomial(F, Esnf, 1)
#     return AD_draw/F

# """
# CUMULATIVE FREQUENCY DISTRIBUTION GENERATION AND SAMPLING
# """

# def getMultipleT1T2SamplesFromTiCFD(numberOfSamples, burstDuration, T_Vals, T_CFD, seed=None):
#     """
#     See getT1T2SampleFromTiCFD
#     """
#     if seed is not None:
#         np.random.seed(seed)
    
#     sampleTPairs = []
#     #for i in range(numberOfSamples):
#     _MC_samples = np.random.default_rng().random(numberOfSamples)
#     for MC in _MC_samples:
#         _T1, _T2 = getT1T2SampleFromTiCFD(MC, burstDuration, T_Vals, T_CFD)
#         sampleTPairs.append([_T1, _T2])
#     return sampleTPairs

# def getMultipleT1T2SamplesFromBurstDuration(numberOfSamples, burstDuration, PTwoStateParameters, seed=None):
#     """
#     Generates a CFD from a Time Duration (and k, N values), then uses getMultipleT1T2SampleFromTiCFD to sample from it
#     """
#     # Unpack Two State Kinetic Parameters
#     N = PTwoStateParameters['N']
#     k_1 = PTwoStateParameters['k_1']
#     k_minus1 = PTwoStateParameters['k_minus1']

#     # Generate T1 Distribution Based on this Duration (generation of CFD is limiting step)
#     P_T1_distributionForBurstLength = PTi_TwoState_at_del_t(N=N, burstDuration=burstDuration, k_1=k_1, k_minus1=k_minus1, cumsum=True, debug=False)
#     P_T1s = np.array(P_T1_distributionForBurstLength['T1'])
#     P_T1_cumsum = np.array(P_T1_distributionForBurstLength['cumsum'])

#     return getMultipleT1T2SamplesFromTiCFD(numberOfSamples, burstDuration, T_Vals=P_T1s, T_CFD=P_T1_cumsum, seed=seed)
"""
HIGH RESOLUTION SEARCHING (BASED ON INDIVIDUAL BURST DURATIONS)
"""

"""
COARSE SEARCHING BASED ON BINNING BURST DURATIONS
"""
def getMultipleT1T2SamplesFromBurstDurationHistogram(overSampleFactorK, 
                                                     burstDurationBinValues, 
                                                     burstDurationBinCounts, 
                                                     PTwoStateParameters, 
                                                     seed=None):
    """
    Fast method for generating a simulated T1, T2s
    Takes binned burst duration values and generates CFD only for bin values rather than every burst
    Scales sampling of each CFD based on counts within each bin
    returns dictionary of with keys being burstDurations
    """
    if seed is not None:
        np.random.seed(seed)

    allPairs = {}
    for burstDuration, burstCount in zip(burstDurationBinValues, burstDurationBinCounts):
        scaledSampleFactor = overSampleFactorK*burstCount
        TPairs = KM2S.getMultipleT1T2SamplesFromBurstDuration(scaledSampleFactor, burstDuration, PTwoStateParameters, seed=None)
        allPairs[burstDuration] = {'TPairs': TPairs, 'binCount':burstCount}
    return allPairs

def getF_andT_valuesForDurationBins(burstDataFrame, durationBins, binCentres, durationHist):
    """
    For each time bin, gathers a list of:
        all F values (total flourescence)
            F values used in binomial sampling of E shot-noise
        all duration values

    F values required for binomial sampling
    
    returns dictionary of dictionaries with {E_bin_Centre1: {FVals [F1, F2, ... Fx], Durations: [t1, t2, ... tx], ...}
    durationHist provides a sanity check that there are no mismatch counting
    """
    burstDurations = burstDataFrame['duration']
    
    burstBinFValues = {}
    for i, TBin in enumerate(durationBins[:-1]):    
        binCentre = binCentres[i]
        
        # Filter for time bin
        TBinMask = np.logical_and((burstDurations >= TBin), (burstDurations < durationBins[i+1]))
        _TBinDF = burstDataFrame[TBinMask]

        # Total Flourescence of burst ('flourescence considering FRET events only, hence AD and DD, not AA?)
        BSs = list(_TBinDF.DD + _TBinDF.AD)

        # Take integer (float due to background subtraction)
        BSs = [int(bs) for bs in BSs]
        
        # Get T values
        durationVals = list(burstDurations.values[TBinMask])
        
        # Sanity Checks
        assert len(durationVals) == durationHist[i], "Error, count of points within bin does not match duration count!"
        assert len(BSs) == durationHist[i], "Error, count of points within bin does not match BS count!"
        
        burstBinFValues[binCentre] = {'Count': durationHist[i], 'F_values': BSs, 'durationValues': durationVals}
        
    return burstBinFValues

def getEvaluesFromDictionaryOfTPairs(E1, E2, overSampleFactorK, durationBinCentres, simulatedTPairsDict, FvaluesBinDict):
    """
    Importantly requires a dictionary (with the same keys!) of F values for each duration bin, F values are
    what are used to sample from a binomial distribution of acceptor photons
    Note, doesnt matter which E value is paired to each F value as all E values drawn from the same distribution (i.e. all F have same duration)

    Returns dictionary keys are bin values, values are E dictionary
    """
    EvalueDict = {}

    for binCentre in durationBinCentres:
        # Get TPairs and F values for this duration
        TPairs = simulatedTPairsDict[binCentre]['TPairs']
        FValues =  FvaluesBinDict[binCentre]['F_values']

        # Calculate shot-noise free simulated E values
        E_snf = getEsnfForListOfRepeatSimulatedPairs(E1, E2, TPairs)

        # Check sampling dimensions correct otherwise E sampling will fail
        assert overSampleFactorK*len(FValues) == len(TPairs), "k*Ncount does not equal number of E values, sampling dimensions wrong"

        # Calcualte the E shot-noise, with appropraite sampling
        # Note, doesnt matter which E value is paired to each F value as all E values drawn from the same distribution
        E_sn = []
        for i, _F in enumerate(FValues):
            for _E_snf in E_snf[i*overSampleFactorK:((i+1)*overSampleFactorK)]:
                    # Uses fast method which uses optimised numpy binomial avoids generating random number separtely
                    E_sn.append(KM2S.getEShotNoise(_F, _E_snf))

        # Add to Dictionary
        EvalueDict[binCentre] = {'Esnf':E_snf, 'Esn':E_sn}
    return EvalueDict


def getFlatEvaluesFromDictionaryOfTPairs(E1, 
                                         E2,
                                         overSampleFactorK,
                                         durationBinCentres,
                                         simulatedTPairsDict,
                                         FvaluesBinDict):
    """
    Very similar to getEvaluesFromDictionaryOfTPairs but flattens the returned dictionary ready for
    histogram analysis of simulated Es
    """
    EvalueDict = getEvaluesFromDictionaryOfTPairs(E1,
                                                      E2,
                                                      overSampleFactorK,
                                                      durationBinCentres,
                                                      simulatedTPairsDict,
                                                      FvaluesBinDict)

    # Flatten Esnf, and Esn
    sim_Esnf = [e for k, d in EvalueDict.items() for e in d['Esnf']]
    sim_Esn = [e for k, d in EvalueDict.items() for e in d['Esn']]
    
    return sim_Esnf, sim_Esn











"""
SIMULATED E UTILITIES
"""
def getEsnfForListOfRepeatSimulatedPairs(E1, E2, TSimulatedPairs):
    """
    Expects an input of [ [T1, T2], [T1, T2],  ...]
    """
    E_simulated = []
    for Pair in TSimulatedPairs:
        # Unpack
        _T1, _T2 = Pair
        # Calculate
        E_simulated.append(KM2S.E_snf_TwoStatesEqualBrightness(T1=_T1, E1=E1, T2=_T2, E2=E2))
    return E_simulated

def getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK):
    simulated_E_Hist, _ = np.histogram(simulatedEs, Ebins)
    simulated_E_Hist_Scaled = simulated_E_Hist/overSamplingFactorK
    return simulated_E_Hist_Scaled


# # # # # # # # # BREAK # # # # # # # # # #

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

"""
END TO END E HIST COMPARISONS - SANITY CHECKS MAINLY
"""
def endToEndEHistCompare(experimentalObjectsDict,
                      PTwoStateParameters,
                      E1,
                      E2,
                      ShotNoiseFree = False,
                      ShotNoise = True,
                      plothists = True):
    # unpack experimental objects
    overSampleFactorK = experimentalObjectsDict['overSampleFactorK']
    burstDurationBinValues = experimentalObjectsDict['burstDurationBinValues']
    burstDurationBinCounts = experimentalObjectsDict['burstDurationBinCounts']
    seed = experimentalObjectsDict['seed']
    FvaluesBinDict = experimentalObjectsDict['FvaluesBinDict']
    Ebins = experimentalObjectsDict['Ebins']
    EBinCentres = experimentalObjectsDict['EBinCentres']
    ExperimentEHist = experimentalObjectsDict['ExperimentEHist']
    
    # For Each bin simulate a range of T pairs (Soon for a range of ks)
    simulatedTPairsDict = getMultipleT1T2SamplesFromBurstDurationHistogram(overSampleFactorK,
                                                                               burstDurationBinValues,
                                                                               burstDurationBinCounts,
                                                                               PTwoStateParameters,
                                                                               seed=seed)

    # Calculate Esnf, and Esn for each bin
    sim_Esnf, sim_Esn = getFlatEvaluesFromDictionaryOfTPairs(E1,
                                                                 E2,
                                                                 overSampleFactorK,
                                                                 burstDurationBinValues,
                                                                 simulatedTPairsDict,
                                                                 FvaluesBinDict)
    
    # Get Hists
    k1 = PTwoStateParameters['k_1']
    kminus1 = PTwoStateParameters['k_minus1']
    dBin = Ebins[1]-Ebins[0]
    if ShotNoise:
        SSE_sn, scaledSimulatedEsnHist = getSSEAndHistFromListOfEs(sim_Esn, Ebins, overSampleFactorK, ExperimentEHist)
        if plothists:
            plt.bar(EBinCentres, ExperimentEHist, width=dBin, alpha =0.5, label='exp')
            plt.bar(EBinCentres, scaledSimulatedEsnHist, width=dBin, alpha =0.5, label='Simulated Esn')
            plt.title(f'Simulated E histogram shot-noise\nk1: {k1}, kminus1: {kminus1}, SSE: {int(SSE_sn)}')
            plt.legend()
            plt.show()
        
    if ShotNoiseFree:
        SSE_snf, scaledSimulatedEsnfHist = getSSEAndHistFromListOfEs(sim_Esnf, Ebins, overSampleFactorK, ExperimentEHist)
        if plothists:
            dBin = EBinCentres[1] - EBinCentres[0]
            plt.bar(EBinCentres, ExperimentEHist, width=dBin, alpha =0.5, label='exp')
            plt.bar(EBinCentres, scaledSimulatedEsnfHist, width=dBin, alpha =0.5, label='Simulated Esnf')
            plt.title(f'Simulated E histogram shot-noise free\nk1: {k1}, kminus1: {kminus1}, SSE: {int(SSE_snf)}')
            plt.legend()
            plt.show()
  
"""
DEPRECIATED
"""

# def getEHistFromSimulatedTPairs(overSampleFactorK, TSimulatedPairs, E1, E2, EBins):
#     """
#     TSimulatedPairs should be a list of list of T pairs: [ [T1, T2], [T1, T2], ...]
#     """
#     # First Calculate Simulated Es
#     _E_simulated = getEsnfForListOfRepeatSimulatedPairs(E1=E1, E2=E2, TSimulatedPairs=TSimulatedPairs)
    
#     # Calculate simulated E Hist and Compare to Experiment
#     EHistSim, _ = np.histogram(_E_simulated, bins = EBins)
#     return EHistSim/overSampleFactorK

# def getSSEFromSimulatedTPairs(overSampleFactorK, TSimulatedPairs, E1, E2, EBins, EHistExperiment):    
#     """
#     TSimulatedPairs should be a list of list of T pairs: [ [[T1, T2], [T1, T2], ...] , ...]
#     """
#     # Check Experimental Histogram same dimensions as generated histogram
#     assert len(EHistExperiment) == len(EBins)-1, "Dimensions of Experimental Histogram and simulated histogram must match!"
    
#     # Get Histogram For these simulated Ts
#     EHistSim = getEHistFromSimulatedTPairs(overSampleFactorK=overSampleFactorK, TSimulatedPairs=TSimulatedPairs, E1=E1, E2=E2, EBins=EBins)
    
#     # Return SSE
#     return np.sum(np.array(np.array(EHistExperiment) - np.array(EHistSim))**2), EHistSim
















"""
Utilitity

Doesn't really belong here, could be anywhere, this is just to tidy up jupyter notebook
"""
def getSamplePointsOverOrdersOfMag(O_StartPower, O_FinalPower, SampleNumber):
    orders_of_mag = [i for i in range(O_StartPower, O_FinalPower+1)]
    
    # Get all Sample Points
    samplePoints = []
    for O in orders_of_mag[:-1]:
        samplePoints.append(np.linspace(10**O, 10**(O+1), SampleNumber))

    # flatten all samples
    return [point for points in samplePoints for point in points]