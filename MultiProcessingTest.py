import TwoStateKineticModel as KM2S
import PDA as PDA
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys
# from multiprocessing import Pool
import multiprocess as mp
sys.path.append('./../')

import modules.modulesPopulationFitting as PF
import modules.modulesCorrectionFactorsAndPlots as MCF


# Read Device Runs (With corrections)
device_data = pd.read_csv(
    './../processed_data/HairpinReferenceRunsWithCorrections.csv')
concs = sorted(set(device_data.NaCl_mM))

# Take data slice, high FRET 2CDE as dont want to filter on this
df_slice = MCF.typical_S_ALEX_FRET_CDE_filter(
    device_data, FRET2CDEmask=20000)
burstDurations_ms = np.array(list(df_slice['duration']))  # *1e3

# Check that burst binning is sensible
durationBins, durationBinCentres, dDuration = PF.getBins(0, 11*1e-3, 100)
durationHist, _ = np.histogram(df_slice['duration'], bins=durationBins)

# E bins
Ebins, EBinCentres, dBin = PF.getBins(-0.1, 1.1, 50)

# Quick Sample to develop function
sample = df_slice[df_slice.NaCl_mM == concs[3]]

# Get Experimental E histogram
exp_E = sample['E3_avg_beta_gamma']
exp_E_Hist, _ = np.histogram(exp_E, Ebins)


# Get E, k starting values (Coarse search should be ran to generate this DF)
kdf = pd.read_csv(".\\csvs\\30_01_2023_NaCl_297_expansion.csv")
SSE_min = kdf.sort_values('SSE', ascending=True).iloc[0, :]
SSE_min_val = SSE_min.SSE
k1min = SSE_min.k_1
kminus1min = SSE_min.k_minus1
E1min = SSE_min.E1
E2min = SSE_min.E2

BURST_DATA = sample
EXP_E_HIST = exp_E_Hist
E1MIN = E1min
E2MIN = E2min
KMIN = 5
NMIN = 50

def poolFunc(ks):
    
    global BURST_DATA
    global EXP_E_HIST
    global E1MIN
    global E2MIN
    global KMIN
    global NMIN

    _k1, _kminus1 = ks
    
    E_snf, E_sn = PDA.getEsnfEsnFromBurstDataFrame(BurstData=BURST_DATA,
                                                E1=E1MIN,
                                                E2=E2MIN,
                                                K=KMIN,
                                                N=NMIN,
                                                k1=_k1,
                                                kminus1=_kminus1)
    
    sse = PDA.getSSEFromListOfEs(E_sn, Ebins, KMIN, exp_E_Hist)
    # return {'k1': _k1, 'kminus1': _kminus1, 'SSE': sse}
    return f'{_k1},{_kminus1},{sse}'

if __name__ == '__main__':

    # Get E, k starting values (Coarse search should be ran to generate this DF)
    kdf = pd.read_csv(".\\csvs\\30_01_2023_NaCl_297_expansion.csv")
    SSE_min = kdf.sort_values('SSE', ascending=True).iloc[0, :]
    SSE_min_val = SSE_min.SSE
    k1min = SSE_min.k_1
    kminus1min = SSE_min.k_minus1
    E1min = SSE_min.E1
    E2min = SSE_min.E2

    print(
        f'NaCl: {concs[3]} mM, k1 min: {k1min:0.2f}, kminus1 min: {kminus1min:0.2f}, SSE: {SSE_min_val:0.2f}')
    
    # Really this shouldn't be square but be based on 
    # the area of SSE exploration
    k1Space = np.linspace(max(1, 300), 4000, 50)
    kminus1Space = np.linspace(max(1, 300), 1000, 50)

    # Create array of arrays of kspace
    kSpaceTotal = [[_k1, _kminus1] for _k1 in k1Space for _kminus1 in kminus1Space]
    
    with open('mp_csv_test.csv', 'a+') as f:
        f.writelines(f'k1,kminus1,sse')
    
        with mp.Pool(20) as p:
            
            # res = list(p.map(poolFunc, kSpaceTotal))
            # for r in res:
            #     print(r)
            
            # resDF = pd.DataFrame(res)
            #resDF.to_csv('./mpTest3.csv', index=False)
            for result in p.imap_unordered(poolFunc, kSpaceTotal):
                print(result)
                f.writelines(f'{result}\n')