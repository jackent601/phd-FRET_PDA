import numpy as np
import pandas as pd
import TwoStateKineticModel as KM2S

# ================================================================================================================================
#   PROBABILITY DISTRIBUTION ANALYSIS
# ================================================================================================================================

# Method described here: https://pubmed.ncbi.nlm.nih.gov/20575136/

# Takes data from FRET experiments and determines values for E1, E2, k1, kminus1 
# assuming a two state kinetic model (TwoStateKineticModel.py in directory)

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

def getEsnfEsnFromDurationsAndFs(E1, E2, durations, Fs, K, N, k1, kminus1, seed=None):
    """
    Calculates Esnf, Esn for a list of durations with flourescences
    """

    # Get arrays of T1, and T2 from CFD distributions
    allT1s, allT2s = KM2S.p_getT1T2SamplesFromMultipleBurstDurations(durations, K, {'N': N, 'k_1': k1, 'k_minus1': kminus1}, seed=seed)
    
    # Scale Flourescence by oversampling factor for binomial sampling
    F_scaled = Fs.repeat(K)

    # Calculate shot-noise free simulated E values 
    E_snf = KM2S.E_snf_TwoStatesEqualBrightness(allT1s, E1, allT2s, E2)

    # Calculate shot-noise-dependent for each T pair in this duration sample
    E_sn = np.array([KM2S.getEShotNoise(_F, _E_snf) for _F, _E_snf in zip(F_scaled, E_snf)]).ravel() 
    
    return E_snf, E_sn

def getEsnfEsnFromBurstDataFrame(BurstData, E1, E2, K, N, k1, kminus1, seed=None):
    d, F = getDurationsAndFlourescence(BurstData)
    return getEsnfEsnFromDurationsAndFs(E1=E1, E2=E2, durations=d, Fs=F, K=K, N=N, k1=k1, kminus1=kminus1, seed=seed)


# ================================================================================================================================
# SIMULATED E UTILITIES
# ================================================================================================================================

def getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK):
    simulated_E_Hist, _ = np.histogram(simulatedEs, Ebins)
    simulated_E_Hist_Scaled = simulated_E_Hist/overSamplingFactorK
    return simulated_E_Hist_Scaled

def getSSEAndHistFromListOfEs(simulatedEs, Ebins, overSamplingFactorK, experimentEHist):
    """
    Ultimately parameter determination is based on comparing the 'simulated' E hist to experimental E hist
    """
    # Get Hist
    simulated_E_Hist_Scaled = getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK)
    # Return SSE
    return np.sum(np.array(np.array(experimentEHist) - np.array(simulated_E_Hist_Scaled))**2), simulated_E_Hist_Scaled

def getSSEFromListOfEs(simulatedEs, Ebins, overSamplingFactorK, experimentEHist):
    """
    Identical to getSSEAndHistFromListOfEs but quicker as doesnt return hist object, only comparison SSE metric
    """
    # Get Hist
    simulated_E_Hist_Scaled = getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK)
    # Return SSE
    return np.sum(np.array(np.array(experimentEHist) - np.array(simulated_E_Hist_Scaled))**2)

