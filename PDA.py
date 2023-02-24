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

# ================================================================================================================================
# BINOMIAL SPREADING FUNCTIONS 
# ================================================================================================================================
def getEsn_FromEsnf_AndF(F, Esnf, EsnfWeights, Ebins):
    """
    Binomially spreads Esnf values to obtain Esn (shot-noise)
    EsnfWeights is used to weight each binomial distribution for original likelyhood of Esnf value
    Ebins needed to find final weights within bins
    """
    # Get Binomial distribution for each Esnf_i (centre)
    EsnfBinomials = scipy.stats.binom(F, Esnf) 
    
    # Also calculate the corresponding Esn for each potential value of of acceptor flourescence the 
    # above distributions describe
    Esn = np.array(range(0, F+1))/F
    
    # Calculates the p(Esn) for each Fr (Red photon) in F (at a particular Esnf_i) 
    # Note where Fr > F p(Esn) = 0 so appropriate to loop in this manner
    EsnBinomials = np.zeros((F+1, len(Esnf)))
    for f in range(0, F+1):
        EsnBinomials[f] = EsnfBinomials.pmf(f)

    # returns an array where rows are Fg and columns are Esnf_i 
    
    # Now need to weight each Esn binomial with original likelyhood of Esnf (from gaussian)
    # Each column is a E_snf so matrix multiple by array of p(E_snf)
    pWeightedEsn_i = EsnBinomials*EsnfWeights
    
    # Sum each weighted Esn for final p(Esn)
    pEsn = np.nansum(pWeightedEsn_i, axis=1)
    
    # Finally get the binned probability to add together different bursts
    discrete_pEsn, _, _ = scipy.stats.binned_statistic(Esn, pEsn, statistic='sum', bins=Ebins)
    
    return discrete_pEsn

# ================================================================================================================================
# GAUSSIAN SPREADING FUNCTIONS 
# ================================================================================================================================
def calcR(E, r0):
    """From FRET Definition, used to add gaussian noise"""
    return r0*((1/E)-1)**(1/6)

def calcE_from_r(r, r0):
    """From FRET Definition, used to add gaussian noise"""
    return 1/(1+(r/r0)**6)

def getGaussianObjects(E1_0, E2_0, r0, rDeviation, gaussianResolution):
    """
    The gaussian noise parameters added (to account for experimental spread always being greater than that of shot-noise) only needs calculated once per
    experiment, hence to speed up processing the 'objects' used are passed into functions as parameters rather than re-calculated on the fly
    """
    # Get rs
    r1 = calcR(E1_0, r0)
    r2 = calcR(E2_0, r0)

    # Get distribution of rs
    r1Norm = scipy.stats.norm(loc=r1, scale=rDeviation)
    r2Norm = scipy.stats.norm(loc=r2, scale=rDeviation)

    r1vals = np.linspace(r1Norm.ppf(0.01), r1Norm.ppf(0.99), gaussianResolution)
    r2vals = np.linspace(r2Norm.ppf(0.01), r2Norm.ppf(0.99), gaussianResolution)

    r1pdf = r1Norm.pdf(r1vals)/sum(r1Norm.pdf(r1vals)) # Normalising continuous -> Discrete
    r2pdf = r2Norm.pdf(r2vals)/sum(r2Norm.pdf(r2vals)) # Normalising continuous -> Discrete

    # Evals have the 'resolution' of gaussianResolution
    E1GaussianVals = calcE_from_r(r1vals, r0)
    E2GaussianVals = calcE_from_r(r2vals, r0)

    # Calculate the 2D distribution in E1, E2
    # this give (row, column) : (E1, E2), important for calculating across t values (explained in jupyter notebook 'gaussian')
    #    --- E2 ---
    #  | 
    # E1
    #  |
    E1E2Gaussian2Dpdf = r1pdf[:, np.newaxis]*r2pdf
    return E1GaussianVals, E2GaussianVals, E1E2Gaussian2Dpdf

def getGaussianpEsnf_forBurst(PT1s, T1s, T2s, E1GaussianVals, E2GaussianVals, E1E2Gaussian2Dpdf, gaussianResolution, Ebins):
    """
    Using gaussian spreading 'objects' (see above) spreads Esnf values using gaussian processing
    involves lots of matrix multiplication - se jupyter notebook 'gaussian' to walk through this processing
    """
    # Get all t1*E1s
    # t1_Transpose * E1 => len(t) x len(E): (x,y)
    t1E1 = T1s[:, np.newaxis]*E1GaussianVals

    # Get all t2*E2s
    # t2Transpose * E2 => 3x5 (x,y)
    t2E2 = T2s[:, np.newaxis]*E2GaussianVals

    # tile t1E1 in z: (slice, (row, column)) : (t, (t*E1, '*'))
    t1E13Dz = np.repeat(t1E1[:, :, np.newaxis], repeats = gaussianResolution, axis=2)

    # tile t2E2 in z: (slice, (row, column)) : (t, ('*', t*E1))
    t2E23Dz = np.repeat(t2E2[:, np.newaxis, :], repeats = gaussianResolution, axis=1)

    # Summing these 3D matrices gives t1E1 + t2E2 for all values of E1, E2, t1, t2 
    # (slice, (row, column)) : (t, (t*E1, t*E2))
    t1E1_plus_t2E2 = t1E13Dz + t2E23Dz

    # Each (row, column) slice maps to the SAME 2D probability distribution p(E1, E2): (row, column) : (E1, E2), so need to extend this in x by len(t)
    # (t, (E1, E2))
    pE1E2z = np.repeat(E1E2Gaussian2Dpdf[np.newaxis, :, :], repeats = len(T1s), axis=0)

    # Each slice (t) needs scaled by _original_ probability of that t, p(t)
    pScaled_pE1E2z = PT1s[:, np.newaxis, np.newaxis]*pE1E2z

    # So we now have a 3D matrix for all E1, E2, t values, and a MAPPED 3D matrix for the scaled probability of that E1, E2, t combination
    # Flattening each 3D matrix and binning yields the final p(Esnf)!t1E1_plus_t2E2.flatten()
    pEsnf_Gauss, _, _ = scipy.stats.binned_statistic(x=t1E1_plus_t2E2.flatten(), values=pScaled_pE1E2z.flatten(), statistic='sum', bins=Ebins)
    
    return pEsnf_Gauss
    
# ================================================================================================================================
# BURST FUNCTIONS
# ================================================================================================================================

def getDiscrete_pEsn_forBurst_withGaussian(N, Ebins, EbinCentres, F, burstDuration, k1, kminus1, E1, E2, E1GaussianVals, E2GaussianVals, E1E2Gaussian2Dpdf, gaussianResolution):
    # Get p(T1) for this burst duration
    T1s, T2s, PT1s = KM2S.PTi_TwoState_at_del_t(
        N, burstDuration, k1, kminus1, cumsum=False, debug=False)
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # Get Esnf with Gaussian Noise (This bins E values TODO - Device what resolution effects this has...)
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    pEsnf_Gauss = getGaussianpEsnf_forBurst(PT1s=PT1s,
                                            T1s=T1s,
                                            T2s=T2s, 
                                            E1GaussianVals=E1GaussianVals, 
                                            E2GaussianVals=E2GaussianVals, 
                                            E1E2Gaussian2Dpdf=E1E2Gaussian2Dpdf, 
                                            gaussianResolution=gaussianResolution, 
                                            Ebins=Ebins)
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # Get final Esn by binomial spreading
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    discrete_pEsn = getEsn_FromEsnf_AndF(F=F, Esnf=EbinCentres, EsnfWeights=pEsnf_Gauss, Ebins=Ebins)
    
    return discrete_pEsn

def getDiscrete_pEsn_forBurst(N, Ebins, F, burstDuration, k1, kminus1, E1, E2):
    """
    OLD - no gaussian spreading
    """
    # Get p(T1) for this burst duration
    T1s, T2s, PT1s = KM2S.PTi_TwoState_at_del_t(
        N, burstDuration, k1, kminus1, cumsum=False, debug=False)

    # get E snf corresponding to T1 values, note p(Esnf) = p(T1)
    Esnf = KM2S.E_snf_TwoStatesEqualBrightness(T1s, E1, T2s, E2)
    
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # Get final Esn by binomial spreading
    # = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    discrete_pEsn = getEsn_FromEsnf_AndF(F=F, Esnf=Esnf, EsnfWeights=PT1s, Ebins=Ebins)
    
    return discrete_pEsn

# ================================================================================================================================
# MULTIPLE BURST FUNCTIONS
# ================================================================================================================================

def getDiscrete_pEsn_forMultipleBurst_withGaussian(N, Ebins, EbinCentres, F, burstDurations, k1, kminus1, E1, E2, r0, rDeviation, gaussianResolution, Weights=None):
    # TODO - provide gaussian parameters as dictionary
    # TODO - pass experiemental params as dictionary
    # Can either sum as you go (save space? at cost of time?) or sum at end (save time? at cost of space?)
    # Weights provide scaling for each duration in case of binned sampling
    """
    F: array
    burstDurations: array
    """
    if Weights is None:
        Weights = np.full(len(burstDurations), 1)

    # Calculate Gaussian Parameters (saves repeated calculation)
    E1GaussianVals, E2GaussianVals, E1E2Gaussian2Dpdf = getGaussianObjects(
        E1_0=E1, E2_0=E2, r0=r0, rDeviation=rDeviation, gaussianResolution=gaussianResolution)

    bursts_discrete_pEsn = []
    for b, _F, W in zip(burstDurations, F, Weights):
        bursts_discrete_pEsn.append(W*getDiscrete_pEsn_forBurst_withGaussian(N=N, 
                                                                             Ebins=Ebins, 
                                                                             EbinCentres=EbinCentres, 
                                                                             F=_F, 
                                                                             burstDuration=b, 
                                                                             k1=k1, 
                                                                             kminus1=kminus1, 
                                                                             E1=E1,
                                                                             E2=E2, 
                                                                             E1GaussianVals=E1GaussianVals, 
                                                                             E2GaussianVals=E2GaussianVals, 
                                                                             E1E2Gaussian2Dpdf=E1E2Gaussian2Dpdf,
                                                                             gaussianResolution=gaussianResolution))
    discrete_pEsn = np.array(bursts_discrete_pEsn).sum(axis=0)
    return discrete_pEsn

def getDiscrete_pEsn_forMultipleBurst(N, Ebins, F, burstDurations, k1, kminus1, E1, E2, Weights=None):
    # Can either sum as you go (save space? at cost of time?) or sum at end (save time? at cost of space?)
    # Weights provide scaling for each duration in case of binned sampling
    if Weights is None:
        Weights = np.full(len(burstDurations), 1)
    
    bursts_discrete_pEsn = []
    for b, _F, W in zip(burstDurations, F, Weights):
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
# DATAFRAME FUNCTIONS
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

# FAST FINDING (first bins bursts to reduce cdf generation burden)
def fastFind_FandBs_fromBurstDataFrame(BurstData, nBurstBins, Ebins):
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
    
    return tsBinnedNonZero, tBinCentresNonZero, FmeanIntsNonZero
 
# WITH GAUSSIAN SPREADING
def get_pEsn_fromBurstDataFrame_withGaussian(BurstData, Ebins, EbinCentres, E1, E2, N, k1, kminus1, r0, rDeviation, gaussianResolution):
    d, F = getDurationsAndFlourescence(BurstData)
    return getDiscrete_pEsn_forMultipleBurst_withGaussian(N, Ebins, EbinCentres, F, d, k1, kminus1, E1, E2, r0, rDeviation, gaussianResolution)

def fastFindpEsn_fromBurstDataFrame_withGaussian(BurstData, nBurstBins, Ebins, EbinCentres, E1, E2, N, k1, kminus1, r0, rDeviation, gaussianResolution):
    """
    Fast find for rough searching, bins duration values and runs average t F within each bin scaling resulting
    pEsn for number of bursts within that bin
    """
    # 'fast' - Get Durations and Flourescences by Binning duration values
    tsBinnedNonZero, tBinCentresNonZero, FmeanIntsNonZero = fastFind_FandBs_fromBurstDataFrame(BurstData, nBurstBins, Ebins)

    # Modified 'getDiscrete_pEsn_forMultipleBurst' to scale for counts within each bin
    discrete_pEsn = getDiscrete_pEsn_forMultipleBurst_withGaussian(N, 
                                                                   Ebins,
                                                                   EbinCentres,
                                                                   FmeanIntsNonZero,
                                                                   tBinCentresNonZero,
                                                                   k1,
                                                                   kminus1,
                                                                   E1,
                                                                   E2,
                                                                   r0,
                                                                   rDeviation,
                                                                   gaussianResolution,
                                                                   Weights=tsBinnedNonZero)
    return discrete_pEsn

# WITHOUT GAUSSIAN SPREADING
def get_pEsn_fromBurstDataFrame(BurstData, Ebins, E1, E2, N, k1, kminus1):
    d, F = getDurationsAndFlourescence(BurstData)
    return getDiscrete_pEsn_forMultipleBurst(N, Ebins, F, d, k1, kminus1, E1, E2)

def fastFindpEsn_fromBurstDataFrame(BurstData, nBurstBins, Ebins, E1, E2, N, k1, kminus1):
    """
    Fast find for rough searching, bins duration values and runs average t F within each bin scaling resulting
    pEsn for number of bursts within that bin
    """
    # 'fast' - Get Durations and Flourescences by Binning duration values
    tsBinnedNonZero, tBinCentresNonZero, FmeanIntsNonZero = fastFind_FandBs_fromBurstDataFrame(BurstData, nBurstBins, Ebins)

    # Modified 'getDiscrete_pEsn_forMultipleBurst' to scale for counts within each bin
    discrete_pEsn = getDiscrete_pEsn_forMultipleBurst(N=N,
                                                      Ebins=Ebins,
                                                      F=FmeanIntsNonZero,
                                                      burstDurations=tBinCentresNonZero,
                                                      k1=k1, 
                                                      kminus1=kminus1,
                                                      E1=E1,
                                                      E2=E2,
                                                      Weights=tsBinnedNonZero)
    return discrete_pEsn

# ================================================================================================================================
# HIGH RESOLUTION SEARCHING (BASED ON INDIVIDUAL BURST DURATIONS)
# ================================================================================================================================


# def getEsnfEsnFromDurationsAndFs(E1, E2, durations, Fs, K, N, k1, kminus1, seed=None):
#     """
#     Calculates Esnf, Esn for a list of durations with flourescences
#     """

#     # Get arrays of T1, and T2 from CFD distributions
#     allT1s, allT2s = KM2S.p_getT1T2SamplesFromMultipleBurstDurations(durations, K, {'N': N, 'k_1': k1, 'k_minus1': kminus1}, seed=seed)
    
#     # Scale Flourescence by oversampling factor for binomial sampling
#     F_scaled = Fs.repeat(K)

#     # Calculate shot-noise free simulated E values 
#     E_snf = KM2S.E_snf_TwoStatesEqualBrightness(allT1s, E1, allT2s, E2)

#     # Calculate shot-noise-dependent for each T pair in this duration sample
#     E_sn = np.array([KM2S.getEShotNoise(_F, _E_snf) for _F, _E_snf in zip(F_scaled, E_snf)]).ravel() 
    
#     return E_snf, E_sn

# def getEsnfEsnFromBurstDataFrame(BurstData, E1, E2, K, N, k1, kminus1, seed=None):
#     d, F = getDurationsAndFlourescence(BurstData)
#     return getEsnfEsnFromDurationsAndFs(E1=E1, E2=E2, durations=d, Fs=F, K=K, N=N, k1=k1, kminus1=kminus1, seed=seed)













# ================================================================================================================================
# SIMULATED E UTILITIES
# ================================================================================================================================

# def getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK):
#     # OLD FUNCTION
#     simulated_E_Hist, _ = np.histogram(simulatedEs, Ebins)
#     simulated_E_Hist_Scaled = simulated_E_Hist/overSamplingFactorK
#     return simulated_E_Hist_Scaled

# def getSSEAndHistFromListOfEs(simulatedEs, Ebins, overSamplingFactorK, experimentEHist):
#     """
#     Ultimately parameter determination is based on comparing the 'simulated' E hist to experimental E hist
#     """
#     # Get Hist
#     simulated_E_Hist_Scaled = getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK)
#     # Return SSE
#     return np.sum(np.array(np.array(experimentEHist) - np.array(simulated_E_Hist_Scaled))**2), simulated_E_Hist_Scaled

# def getSSEFromListOfEs(simulatedEs, Ebins, overSamplingFactorK, experimentEHist):
#     """
#     Identical to getSSEAndHistFromListOfEs but quicker as doesnt return hist object, only comparison SSE metric
#     """
#     # Get Hist
#     simulated_E_Hist_Scaled = getScaledSimulatedEHist(simulatedEs, Ebins, overSamplingFactorK)
#     # Return SSE
#     return np.sum(np.array(np.array(experimentEHist) - np.array(simulated_E_Hist_Scaled))**2)

# def getSSEFromEHists(simulatedEHist, experimentEHist):
#     return np.sum(np.array(np.array(experimentEHist) - np.array(simulatedEHist))**2)

# ================================================================================================================================
# GRADIENT DESCENT - OPTIMISES k1, kminus1 VALUES
# ================================================================================================================================
