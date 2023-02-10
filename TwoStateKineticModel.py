from scipy.special import i0, i1
from scipy.stats import binom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    bessel_arg = 2*np.sqrt(k1*k2*T1*T2)
    
    coeff1 = (2*k1*k2)/(k1+k2)
    bessel1 = i0(bessel_arg)
    
    coeff2 = ((k2*T1+k1*T2)/(k1+k2))*((np.sqrt(k1*k2))/(np.sqrt(T1*T2)))
    bessel2 = i1(bessel_arg)
    
    exp_term = np.exp(-k1*T1-k2*T2)*del_T
    
    return (coeff1*bessel1 + coeff2*bessel2)*exp_term

def PTi_TwoState_at_del_t(N, burstDuration, k_1, k_minus1, cumsum = True, debug = False):
    """
    For given burst duration, k_1, k_minus1 calculates the probability distribution of spending t seconds in state 1 and state 2
    Returns a dataframe of probability distribution for delt
    """
    
    del_T = burstDuration/N
    
    P_T1s = []
    
    for n in range(N+1):

        if n == 0:
            # Primarily in State E2
            _T1 = del_T/4
            _T2 = burstDuration - _T1
            _P_T1 = P_TwoState_T1_0(k1=k_1, k2=k_minus1, del_t=burstDuration) + P_TwoState_T1(k1=k_1, k2=k_minus1, T1=_T1, T2=_T2, del_T=del_T/2)
        elif n == N:
            # Primarily in State E1
            _T2 = del_T/4
            _T1 = burstDuration - _T2
            _P_T1 = P_TwoState_T2_0(k1=k_1, k2=k_minus1, del_t=burstDuration) + P_TwoState_T1(k1=k_1, k2=k_minus1, T1=_T1, T2=_T2, del_T=del_T/2)
        else:
            # Transition Between States
            _T1 = n*del_T
            _T2 = burstDuration - _T1
            _P_T1 = P_TwoState_T1(k1=k_1, k2=k_minus1, T1=_T1, T2=_T2, del_T=del_T)

        P_T1s.append({'T1':_T1, 'T2':_T2, 'PT1': _P_T1, 'i': n})
                
        if debug:
            print(f'T1: {_T1*1e3:0.2f} ms, T2: {_T2*1e3:0.2f} ms, P_T1: {_P_T1}')
        
    if np.any(np.array([d['PT1'] for d in P_T1s]) < 0):
            print("WARNING NEGATIVE PROBABILITIES ENCOUNTERED")
        
    P_T1s_df = pd.DataFrame(P_T1s)
    
    if cumsum:
        P_T1s_df['cumsum'] = P_T1s_df.PT1.cumsum()
            
    return P_T1s_df

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
    P_T1_distributionForBurstLength = PTi_TwoState_at_del_t(N=N, burstDuration=burstDuration, k_1=k_1, k_minus1=k_minus1, cumsum=True, debug=False)
    P_T1s = np.array(P_T1_distributionForBurstLength['T1'])
    P_T1_cumsum = np.array(P_T1_distributionForBurstLength['cumsum'])
    return getMultipleT1T2SamplesFromTiCFD(numberOfSamples, burstDuration, T_Vals=P_T1s, T_CFD=P_T1_cumsum, seed=seed)

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