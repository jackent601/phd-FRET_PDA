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

# NUMBER_RAN = 0

def runPDAOverkSpace_MultiProcess(ks, PROCESSOR_COUNT, outputPath, burstData, expEHist, EBins, E1min, E2min, K, N):
    """
    
    ks should be an array of k pairs [ [k1_1, kminus1_1], [k1_2, kminus1_2], .. [,] ]
    """
    NUMBER_RAN = 0
    kSpaceSize = len(ks)
    
    # Make Global Initialiser so data is available in each child process
    def makeDataGlobal(burstData, expEHist, EBins, E1min, E2min, K, N):
        # Declare variables to be Global
        global BURST_DATA
        global EXP_E_HIST
        global E_BINS
        global E1MIN
        global E2MIN
        global K0
        global N0
        
        # Update 'global_cache' with a value, now *implicitly* accessible in func
        BURST_DATA = burstData
        EXP_E_HIST = expEHist
        E_BINS = EBins
        E1MIN = E1min
        E2MIN = E2min
        K0 = K
        N0 = N
        
    
    def poolFunc_getSSEForkpair(ks):
        _k1, _kminus1 = ks
        
        E_snf, E_sn = PDA.getEsnfEsnFromBurstDataFrame(BurstData=BURST_DATA,
                                                       E1=E1MIN,
                                                       E2=E2MIN,
                                                       K=K0,
                                                       N=N0,
                                                       k1=_k1,
                                                       kminus1=_kminus1)

        sse = PDA.getSSEFromListOfEs(E_sn, E_BINS, K0, EXP_E_HIST)
        # return {'k1': _k1, 'kminus1': _kminus1, 'SSE': sse}
        return f'{_k1},{_kminus1},{sse:0.2f}'
    
    
    with open(outputPath, 'a+') as f:
        f.writelines(f'k1,kminus1,SSE\n')
    
        with mp.Pool(PROCESSOR_COUNT, initializer=makeDataGlobal, initargs=(burstData, expEHist, EBins, E1min, E2min, K, N,)) as p:
            for result in p.imap_unordered(poolFunc_getSSEForkpair, kSpaceTotal):
                r_string = result
                print(f'{100*NUMBER_RAN/kSpaceSize:0.2F} % Ran \t {r_string}')
                f.writelines(f'{result}\n')
                NUMBER_RAN += 1
    

if __name__ == '__main__':
    # =======================================================================================================================
    # TODO - Include Parser to functionalise this code
    # =======================================================================================================================
    
    # Read Device Runs (With corrections)
    device_data = pd.read_csv('./../processed_data/HairpinReferenceRunsWithCorrections.csv')
    concs = sorted(set(device_data.NaCl_mM))

    # Take data slice, high FRET 2CDE as dont want to filter on this
    df_slice = MCF.typical_S_ALEX_FRET_CDE_filter(device_data, FRET2CDEmask=20000)
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
    
    # # Get E, k starting values (Coarse search should be ran to generate this DF)
    kdf = pd.read_csv(".\\csvs\\30_01_2023_NaCl_198_expansion.csv")
    SSE_min = kdf.sort_values('SSE', ascending=True).iloc[0, :]
    SSE_min_val = SSE_min.SSE
    k1min = SSE_min.k_1
    kminus1min = SSE_min.k_minus1
    E1min = SSE_min.E1
    E2min = SSE_min.E2
    
    k1Space = np.linspace(max(1, 300), 4000, 5)
    kminus1Space = np.linspace(max(1, 300), 1000, 5)
    kSpaceTotal = [[_k1, _kminus1] for _k1 in k1Space for _kminus1 in kminus1Space]
    
    runPDAOverkSpace_MultiProcess(ks=kSpaceTotal, 
                                  PROCESSOR_COUNT=20,
                                  outputPath='./mpFuncTest.csv',
                                  burstData=sample,
                                  expEHist=exp_E_Hist,
                                  EBins=Ebins,
                                  E1min=E1min,
                                  E2min=E2min,
                                  K=5,
                                  N=50)
        
    # # Make Global Initialiser so data is available in each child process
    # def makeDataGlobal(burstData, expEHist, EBins, E1min, E2min, K, N):
    #     # Declare variables to be Global
    #     global BURST_DATA
    #     global EXP_E_HIST
    #     global E_BINS
    #     global E1MIN
    #     global E2MIN
    #     global K0
    #     global N0
        
    #     # Update 'global_cache' with a value, now *implicitly* accessible in func
    #     BURST_DATA = burstData
    #     EXP_E_HIST = expEHist
    #     E_BINS = EBins
    #     E1MIN = E1min
    #     E2MIN = E2min
    #     K0 = K
    #     N0 = N
        
    
    # def poolFunc(ks):
    #     _k1, _kminus1 = ks
        
    #     E_snf, E_sn = PDA.getEsnfEsnFromBurstDataFrame(BurstData=BURST_DATA,
    #                                                    E1=E1MIN,
    #                                                    E2=E2MIN,
    #                                                    K=K0,
    #                                                    N=N0,
    #                                                    k1=_k1,
    #                                                    kminus1=_kminus1)

    #     sse = PDA.getSSEFromListOfEs(E_sn, E_BINS, K0, EXP_E_HIST)
    #     # return {'k1': _k1, 'kminus1': _kminus1, 'SSE': sse}
    #     return f'{_k1},{_kminus1},{sse:0.2f}'

    # # Get E, k starting values (Coarse search should be ran to generate this DF)
    # kdf = pd.read_csv(".\\csvs\\30_01_2023_NaCl_198_expansion.csv")
    # SSE_min = kdf.sort_values('SSE', ascending=True).iloc[0, :]
    # SSE_min_val = SSE_min.SSE
    # k1min = SSE_min.k_1
    # kminus1min = SSE_min.k_minus1
    # E1min = SSE_min.E1
    # E2min = SSE_min.E2

    # print(
    #     f'NaCl: {concs[3]} mM, k1 min: {k1min:0.2f}, kminus1 min: {kminus1min:0.2f}, SSE: {SSE_min_val:0.2f}')
    
    # # Really this shouldn't be square but be based on 
    # # the area of SSE exploration
    # k1Space = np.linspace(max(1, 300), 4000, 5)
    # kminus1Space = np.linspace(max(1, 300), 1000, 5)

    # # Create array of arrays of kspace
    # kSpaceTotal = [[_k1, _kminus1] for _k1 in k1Space for _kminus1 in kminus1Space]
    # kSpaceLength = len(kSpaceTotal)
    
    # with open('mp_csv_test2.csv', 'a+') as f:
    #     f.writelines(f'k1,kminus1,sse')
    
    #     with mp.Pool(20, initializer=makeDataGlobal, initargs=(sample, exp_E_Hist, Ebins, E1min, E2min, 5, 50,)) as p:
    #         for result in p.imap_unordered(poolFunc, kSpaceTotal):
    #             r_string = result
    #             print(f'{100*NUMBER_RAN/kSpaceLength:0.2F} % Ran \t {r_string}')
    #             f.writelines(f'{result}\n')
    #             NUMBER_RAN += 1