from scipy.special import i0, i1
import numpy as np
import pandas as pd
import scipy

# ================================================================================================================================
#   T1, T2 PROBABILITY DISTRIBUTION FOR BURST DURATION
# ================================================================================================================================

# Kinetic Model for a system that interconverts between two states E1, E2 with constants k1, kminus1

#     Equations ref: Detection of Structural Dynamics by FRET: A Photon Distribution and Fluorescence Lifetime Analysis of Systems with Multiple States
#     DOI: https://pubs.acs.org/doi/10.1021/jp102156t 

# i0, i1 are MODIFIED bessel functions, first kind, and of 0, 1 order respectively
# Note there is a difference between del_t (burst duration) and del_T (width of probability sample)

# Eqs in the limit of small Delta T, here Delta T = burstDuration/N
# ================================================================================================================================

def P_TwoState_T1_0(k1, k2, del_t):
    return (k1/(k1+k2))*np.exp(-k2*del_t)

def P_TwoState_T2_0(k1, k2, del_t):
    return P_TwoState_T1_0(k1=k2, k2=k1, del_t=del_t)

def P_TwoState_T1(k1, k2, T1, T2, del_T):
    """
    Removed variable assignment, does this help numpy array evaluation?
    """
    # bessel_arg = 2*np.sqrt(k1*k2*T1*T2)
    
    # coeff1 = ((2*k1*k2)/(k1+k2))
    # bessel1 = ((2*k1*k2)/(k1+k2))*i0(bessel_arg)
    
    # # coeff2 = ((k2*T1+k1*T2)/(k1+k2))*((np.sqrt(k1*k2))/(np.sqrt(T1*T2)))
    # bessel2 = ((k2*T1+k1*T2)/(k1+k2))*((np.sqrt(k1*k2))/(np.sqrt(T1*T2)))*i1(bessel_arg)
    
    # exp_term = np.exp(-k1*T1-k2*T2)*del_T
    
    # return (bessel1 + bessel2)*exp_term
    return (((k2*T1+k1*T2)/(k1+k2))*((np.sqrt(k1*k2))/(np.sqrt(T1*T2)))*i1(2*np.sqrt(k1*k2*T1*T2)) + ((2*k1*k2)/(k1+k2))*i0(2*np.sqrt(k1*k2*T1*T2)))*np.exp(-k1*T1-k2*T2)*del_T



def PTi_TwoState_at_del_t(N, burstDuration, k_1, k_minus1, cumsum = True, debug = False):
    """
    For given burst duration, k_1, k_minus1 calculates the probability distribution of spending t seconds in state 1 and state 2
    Returns a dataframe of probability distribution for delt
    """
    
    del_T = burstDuration/N
    
    _T1s = np.zeros(N+1)
    _T2s = np.zeros(N+1)
    _PT1s = np.zeros(N+1)
    
    # Handle T1 = 0
    _T1s[0] = del_T/4
    _T2s[0] = burstDuration - del_T/4
    _PT1s[0] = P_TwoState_T1_0(k1=k_1, k2=k_minus1, del_t=burstDuration) + \
        P_TwoState_T1(k1=k_1, k2=k_minus1, T1=del_T/4, T2=burstDuration - del_T/4, del_T=del_T/2)

    # Handle 0 < T1 < BD
    _TransitionSteps = np.array(range(1, N))
    _TransitionT1s = _TransitionSteps*del_T
    _TransitionT2s = burstDuration - _TransitionT1s
    _TransitionPT1s = P_TwoState_T1(k1=k_1, k2=k_minus1, T1=_TransitionT1s, T2=_TransitionT2s, del_T=del_T)
    
    _T1s[1:N] = _TransitionT1s
    _T2s[1:N] = _TransitionT2s
    _PT1s[1:N] = _TransitionPT1s
            
    # Handle T1 = BD
    _T1s[N] = burstDuration - del_T/4
    _T2s[N] = del_T/4
    _PT1s[N] = P_TwoState_T2_0(k1=k_1, k2=k_minus1, del_t=burstDuration) + \
        P_TwoState_T1(k1=k_1, k2=k_minus1, T1=burstDuration - del_T/4, T2=del_T/4, del_T=del_T/2)
        
    if np.any(_PT1s < 0):
            print("WARNING NEGATIVE PROBABILITIES ENCOUNTERED")
    
    if cumsum:
        _cumsum = np.cumsum(_PT1s)
        return _T1s, _T2s, _PT1s, _cumsum
            
    return _T1s, _T2s, _PT1s

# ================================================================================================================================
# CUMULATIVE FREQUENCY DISTRIBUTION GENERATION AND SAMPLING
# ================================================================================================================================

# Uses the above kinetic model to generate probability distributions from burst durations and samples from them
# in a monte carlo fashion

# ================================================================================================================================
def getT1T2SampleFromTiCFD(MC_sample, burstDuration, T_Vals, T_CFD):
    """
    For a cumulative frequency distribution at a given burst duration (T_CFD) Monte Carlo samples from it to get T1 (T_Vals), T2 values (burstDuration-T1)
    burstDuration must be value used to generate the CFD
    """
    # Catch if MC draw greater than max cumsum (this is due to discrete distribution generation)
    if MC_sample >= max(T_CFD):
        return T_Vals[-1], burstDuration-T_Vals[-1]
    if MC_sample <= min(T_CFD):
        return T_Vals[0], burstDuration-T_Vals[0]

    # Get first location where probability exceeds sample
    cumsumIndex = np.argwhere(T_CFD >= MC_sample)[0, 0]

    # Retrieve corresponding T1 value, hence calculate T2 value
    T1 = T_Vals[cumsumIndex]
    T2 = burstDuration - T1
    return T1, T2

def getMultipleT1T2SamplesFromTiCFD(numberOfSamples, burstDuration, T_Vals, T_CFD, seed=None):
    """
    See getT1T2SampleFromTiCFD
    """
    if seed is not None:
        np.random.seed(seed)
    
    sampleTPairs = []
    #for i in range(numberOfSamples):
    _MC_samples = np.random.default_rng().random(numberOfSamples)
    for MC in _MC_samples:
        _T1, _T2 = getT1T2SampleFromTiCFD(MC, burstDuration, T_Vals, T_CFD)
        sampleTPairs.append([_T1, _T2])
    return sampleTPairs

def getMultipleT1T2SamplesFromBurstDuration(numberOfSamples, burstDuration, N, k_1, k_minus1, seed=None):
    """
    Generates a CFD from a Time Duration (and k, N values), then uses getMultipleT1T2SampleFromTiCFD to sample from it
    """
    # Generate T1 Distribution Based on this Duration (generation of CFD is limiting step)
    _T1s, _T2s, _PT1s, _cumsum = PTi_TwoState_at_del_t(N=N, burstDuration=burstDuration, k_1=k_1, k_minus1=k_minus1, cumsum=True, debug=False)
    return getMultipleT1T2SamplesFromTiCFD(numberOfSamples, burstDuration, T_Vals=_T1s, T_CFD=_cumsum, seed=seed)

def p_getMultipleT1T2SamplesFromBurstDuration(numberOfSamples, burstDuration, PTwoStateParameters, seed=None):
    """
    Identical to getMultipleT1T2SamplesFromBurstDuration but N, k_1, k_minus1 provided in a parameter dictionary to tidy code
    """
    # Unpack Two State Kinetic Parameters
    N = PTwoStateParameters['N']
    k_1 = PTwoStateParameters['k_1']
    k_minus1 = PTwoStateParameters['k_minus1']

    # Generate T1 Distribution Based on this Duration (generation of CFD is limiting step)
    P_T1_distributionForBurstLength = PTi_TwoState_at_del_t(N=N, burstDuration=burstDuration, k_1=k_1, k_minus1=k_minus1, cumsum=True, debug=False)
    P_T1s = np.array(P_T1_distributionForBurstLength['T1'])
    P_T1_cumsum = np.array(P_T1_distributionForBurstLength['cumsum'])

    return getMultipleT1T2SamplesFromTiCFD(numberOfSamples, burstDuration, T_Vals=P_T1s, T_CFD=P_T1_cumsum, seed=seed)

def p_getT1T2SamplesFromMultipleBurstDurations(durations, K, PTwoStateParameters, seed=None):
    """
    Runs through each duration, generates a CFD, samples T1, T2 values from CFD
    N, k_1, k_minus1 provided in a parameter dictionary to tidy code
    Returns two numpy arrays of T1, T2 values
    """
    # Unpack Two State Kinetic Parameters
    N = PTwoStateParameters['N']
    k_1 = PTwoStateParameters['k_1']
    k_minus1 = PTwoStateParameters['k_minus1']

    Tpair_samples = []

    # Get T1, T2 samples
    for _d in durations:
        T_pairs = getMultipleT1T2SamplesFromBurstDuration(K, _d, N, k_1, k_minus1, seed=seed)
        Tpair_samples.append(T_pairs)

    # Get arrays of T1, and T2 for paralleled calculation
    allT1s = np.array([Tpair[0] for TpairSample in Tpair_samples for Tpair in TpairSample])
    allT2s = np.array([Tpair[1] for TpairSample in Tpair_samples for Tpair in TpairSample])
    
    return allT1s, allT2s

# ================================================================================================================================
# E SNF (shot-noise free) AND E SN (shot-noise)
# ================================================================================================================================

# Calculating E values, refered to as 'simulated' Es from T1, T2 distributions (generated by Monte-Carlo sampling)

# ================================================================================================================================
    
def E_snf_TwoStates(Q1, T1, E1, Q2, T2, E2):
    """
    Ref: Detection of Structural Dynamics by FRET: A Photon Distribution and Fluorescence Lifetime Analysis of Systems with Multiple States
    """
    return (Q1*T1*E1 + Q2*T2*E2)/(Q1*T1 + Q2*T2)

def E_snf_TwoStatesEqualBrightness(T1, E1, T2, E2):
    """
    Assumes Equal Brightness Q1 = Q2, need a referance discussing appropriateness
    Q, brightness is the total count rate (in both donor and acceptor detection channels) at a given excitiation intensity,
    at a concentration of one moleucle per observation volume
    """
    return E_snf_TwoStates(Q1=1, T1=T1, E1=E1, Q2=1, T2=T2, E2=E2)

def getEShotNoise(F, Esnf):
    """
    ShotNoise adds binomial noise to sample, see reference papers
    Uses numpy module to sample from binomial which is much faster, but loses tracability on MC draw
    """
    AD_draw = np.random.binomial(F, Esnf, 1)
    return AD_draw/F

# ================================================================================================================================
# PROBABILITY APPROACH, AVOIDING MC SAMPLING
# ================================================================================================================================

# Described here: https://pubs.acs.org/doi/10.1021/jp102156t

# ================================================================================================================================


def getDiscrete_pEsn_forBurst(N, Ebins, F, burstDuration, k_1, k_minus1, E1, E2):

    # Get p(T1) for this burst duration
    T1s, T2s, PT1s = PTi_TwoState_at_del_t(
        N, burstDuration, k_1, k_minus1, cumsum=False, debug=False)

    # get E snf corresponding to T1 values, note p(Esnf) = p(T1)
    Esnf = E_snf_TwoStatesEqualBrightness(T1s, E1, T2s, E2)
    
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

def getDiscrete_pEsn_forMultipleBurst(N, Ebins, F, burstDurations, k_1, k_minus1, E1, E2):
    # Can either sum as you go (save space? at cost of time?) or sum at end (save time? at cost of space?)
    bursts_discrete_pEsn = []
    for b, _F in zip(burstDurations, F):
        bursts_discrete_pEsn.append(getDiscrete_pEsn_forBurst(N=N, 
                                                              Ebins=Ebins, 
                                                              F=_F,
                                                              burstDuration=b,
                                                              k_1=k_1,
                                                              k_minus1=k_minus1,
                                                              E1=E1,
                                                              E2=E2))
    discrete_pEsn = np.array(bursts_discrete_pEsn).sum(axis=0)
    return discrete_pEsn