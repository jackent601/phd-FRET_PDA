import numpy as np
import pandas as pd
import TwoStateKineticModel as KM2S
import scipy

# ================================================================================================================================
#   PROBABILITY DISTRIBUTION ANALYSIS
# ================================================================================================================================

# Method described here: https://pubmed.ncbi.nlm.nih.gov/20575136/

# Takes data from FRET experiments and determines values for E1, E2, k1, kminus1 
# assuming a two state kinetic model (TwoStateKineticModel.py in directory)

# ================================================================================================================================


# ================================================================================================================================
# PROBABILITY APPROACH, AVOIDING MC SAMPLING
# ================================================================================================================================

# Described here: https://pubs.acs.org/doi/10.1021/jp102156t

# ================================================================================================================================


def getDiscrete_pEsn_forBurst(N, Ebins, F, burstDuration, k1, kminus1, E1, E2):

    # Get p(T1) for this burst duration
    T1s, T2s, PT1s = KM2S.PTi_TwoState_at_del_t(
        N, burstDuration, k1, kminus1, cumsum=False, debug=False)

    # get E snf corresponding to T1 values, note p(Esnf) = p(T1)
    Esnf = KM2S.E_snf_TwoStatesEqualBrightness(T1s, E1, T2s, E2)
    
    # Get Binomial distribution for each Esnf_i
    EsnfBinomials = scipy.stats.binom(F, Esnf) # Returns a generator for each Esnf_i at flourescence F
    
    # Also calculate the corresponding Esn for each potential value of of acceptor flourescence the 
    # above distributions describe
    Esn = np.array(range(0, F+1))/F
    
    # Calculates the p(Esn) for each Fr (Red photon) in F (at a particular Esnf_i) 
    # Note where Fr > F p(Esn) = 0 so appropriate to loop in this manner
    EsnBinomials = np.zeros((F+1, len(Esnf)))
    for f in range(0, F+1):
        EsnBinomials[f] = EsnfBinomials.pmf(f)

    # returns an array where rows are Fg and columns are Esnf_i 
    
    # Now need to weight each Esn binomial with original likelyhood of Esnf (p(T1))
    # Each column is a E_snf so matrix multiple by array of p(E_snf)
    pWeightedEsn_i = EsnBinomials*PT1s
    
    # Sum each weighted Esn for final p(Esn)
    pEsn = pWeightedEsn_i.sum(axis=1)
    
    # Finally get the binned probability to add together different bursts
    discrete_pEsn, _, _ = scipy.stats.binned_statistic(Esn, pEsn, statistic='sum', bins=Ebins)
    
    return discrete_pEsn

def getDiscrete_pEsn_forMultipleBurst(N, Ebins, F, burstDurations, k1, kminus1, E1, E2):
    # Can either sum as you go (save space? at cost of time?) or sum at end (save time? at cost of space?)
    bursts_discrete_pEsn = []
    for b, _F in zip(burstDurations, F):
        bursts_discrete_pEsn.append(getDiscrete_pEsn_forBurst(N=N, 
                                                              Ebins=Ebins, 
                                                              F=_F,
                                                              burstDuration=b,
                                                              k1=k1,
                                                              kminus1=kminus1,
                                                              E1=E1,
                                                              E2=E2))
    discrete_pEsn = np.array(bursts_discrete_pEsn).sum(axis=0)
    return discrete_pEsn

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

def get_pEsn_fromBurstDataFrame(BurstData, Ebins, E1, E2, N, k1, kminus1):
    d, F = getDurationsAndFlourescence(BurstData)
    return getDiscrete_pEsn_forMultipleBurst(N, Ebins, F, d, k1, kminus1, E1, E2)

def fastFindpEsn_fromBurstDataFrame(BurstData, nBurstBins, Ebins, E1, E2, N, k1, kminus1):
    """
    Fast find for rough searching, bins duration values and runs average t F within each bin scaling resulting
    pEsn for number of bursts within that bin
    """
    # Get Durations and Flourescences
    d, F = getDurationsAndFlourescence(BurstData)

    # Bin duration values
    tsBinned, tBins = np.histogram(d, bins=nBurstBins)
    dtBin = tBins[1]-tBins[0]
    tBinCentres = tBins[:-1] + dtBin/2

    # Get mean flourescence for each bin (Justification in documentation, guassian broadening limits braodening anyway)
    Fmeans, _, _ = scipy.stats.binned_statistic(x=d, values=F, statistic='mean', bins = tBins)
    FmeansInts = np.array([int(F) if not np.isnan(F) else 0 for F in Fmeans])
    
    # Keep only non-zero bins
    tsBinnedNonZero = tsBinned[tsBinned != 0]
    tBinCentresNonZero = tBinCentres[tsBinned != 0]
    FmeanIntsNonZero = FmeansInts[tsBinned != 0]

    # Modified 'getDiscrete_pEsn_forMultipleBurst' to scale for counts within each bin
    bursts_discrete_pEsn = []
    for b, W, _F in zip(tBinCentresNonZero, tsBinnedNonZero, FmeanIntsNonZero):
        bursts_discrete_pEsn.append(W*getDiscrete_pEsn_forBurst(N=N,
                                                                Ebins=Ebins,
                                                                F=_F,
                                                                burstDuration=b,
                                                                k1=k1,
                                                                kminus1=kminus1,
                                                                E1=E1,
                                                                E2=E2))
    discrete_pEsn = np.array(bursts_discrete_pEsn).sum(axis=0)
    return discrete_pEsn

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

def getSSEFromEHists(simulatedEHist, experimentEHist):
    return np.sum(np.array(np.array(experimentEHist) - np.array(simulatedEHist))**2)

