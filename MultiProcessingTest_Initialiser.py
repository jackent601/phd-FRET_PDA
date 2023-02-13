import TwoStateKineticModel as KM2S
import PDA as PDA
import numpy as np
import pandas as pd
import multiprocess as mp
import sys
import os
sys.path.append('./../')
import modules.modulesPopulationFitting as PF
import modules.modulesCorrectionFactorsAndPlots as MCF

# ==================================================================================================================================
#   PDA MULTIPROCESSING
# ==================================================================================================================================

# Multi-processing implementation of PDA for faster search in k space

# Note using multiprocess, rather than pythons inbuilt as multiprocess is compatible with IPython 
# All child threads need to read burst data, hence need to define globally accessible values,
# See here for explanation: https://stackoverflow.com/questions/52543868/pass-data-to-python-multiprocessing-pool-worker-processes

# ==================================================================================================================================


def makeDataGlobal(burstData, expEHist, EBins, E1min, E2min, K, N):
    """
    Make Global Initialiser so data is available in each child process of multi-process
    See here for explanation: https://stackoverflow.com/questions/52543868/pass-data-to-python-multiprocessing-pool-worker-processes
    """
    # Declare variables to be Global
    global BURST_DATA
    global EXP_E_HIST
    global E_BINS
    global E1MIN
    global E2MIN
    global K0
    global N0
    
    # Update globals with a value, now *implicitly* accessible in func
    BURST_DATA = burstData
    EXP_E_HIST = expEHist
    E_BINS = EBins
    E1MIN = E1min
    E2MIN = E2min
    K0 = K
    N0 = N

def poolFunc_getSSEForkpair(kpair):
    """
    Define pool function for each child process in MP, note it returns k values to keep track as multi-processing can
    execute in arbitrary orders
    """
    _k1, _kminus1 = kpair
    try:
        E_snf, E_sn = PDA.getEsnfEsnFromBurstDataFrame(BurstData=BURST_DATA,
                                                    E1=E1MIN,
                                                    E2=E2MIN,
                                                    K=K0,
                                                    N=N0,
                                                    k1=_k1,
                                                    kminus1=_kminus1)

        sse = PDA.getSSEFromListOfEs(E_sn, E_BINS, K0, EXP_E_HIST)
    except Exception as e:
        sse = None
    return _k1, _kminus1, sse


def runPDAOverkSpace_MultiProcess_FileOutput(kpairs, PROCESSOR_COUNT, outputPath, burstData, expEHist, EBins, E1min, E2min, K, N, debug=True):
    """
    Faster than dictionary as doesn't clog up memory
    kpairs should be an array of k pairs [ [k1_1, kminus1_1], [k1_2, kminus1_2], .. [,] ]
    """
    NUMBER_RAN = 0
    kSpaceSize = len(kpairs)
    
    # Make values Global for child processes
    makeDataGlobal(burstData, expEHist, EBins, E1min, E2min, K, N)  
    
    # Ensure output doesnt exist
    assert not os.path.exists(outputPath), f"{outputPath} already exists!"
    
    # Open output (0 is a 'flush' command to write continuously to file rather than dump at end)
    with open(outputPath, 'a+', 0) as f:
        f.writelines(f'k1,kminus1,SSE\n')
    
        # Run PDA using multi-processing
        with mp.Pool(PROCESSOR_COUNT, initializer=makeDataGlobal, initargs=(burstData, expEHist, EBins, E1min, E2min, K, N,)) as p:
            for result in p.imap_unordered(poolFunc_getSSEForkpair, kpairs):
                # unpack result
                _k1, _kminus1, _sse = result
                result_string = f'{_k1},{_kminus1},{_sse}'
                if debug:
                    print(f'{100*NUMBER_RAN/kSpaceSize:0.2F} % Ran \t k1: {_k1}, kminus1: {_kminus1}, SSE: {_sse:0.2f}')
                f.writelines(f'{result_string}\n')
                NUMBER_RAN += 1
    
def runPDAOverkSpace_MultiProcess_DictListOutput(kpairs, PROCESSOR_COUNT, burstData, expEHist, EBins, E1min, E2min, K, N, debug=True):
    """
    Slower than outputting to file as dictionary list grows in memory
    kpairs should be an array of k pairs [ [k1_1, kminus1_1], [k1_2, kminus1_2], .. [,] ]
    """
    NUMBER_RAN = 0
    kSpaceSize = len(kpairs)
    
    # Make values Global for child processes
    makeDataGlobal(burstData, expEHist, EBins, E1min, E2min, K, N)  
    
    # Run PDA using multi-processing
    resultDictList = []
    with mp.Pool(PROCESSOR_COUNT, initializer=makeDataGlobal, initargs=(burstData, expEHist, EBins, E1min, E2min, K, N,)) as p:
        for result in p.imap_unordered(poolFunc_getSSEForkpair, kpairs):
            # unpack result
            _k1, _kminus1, _sse = result
            resultDictList.append({'k1': _k1, 'kminus1': _kminus1, 'SSE': _sse})
            if debug:
                print(f'{100*NUMBER_RAN/kSpaceSize:0.2F} % Ran \t k1: {_k1}, kminus1: {_kminus1}, SSE: {_sse:0.2f}')
            NUMBER_RAN += 1
    return resultDictList
                
                
if __name__ == '__main__':
    # =======================================================================================================================
    # TODO - Include Parser to functionalise this code
    # =======================================================================================================================
    
    # =======================================================================================================================
    # EXAMPLE
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
    
    # File Output Function
    runPDAOverkSpace_MultiProcess_FileOutput(kpairs=kSpaceTotal,
                                             PROCESSOR_COUNT=20,
                                             outputPath='./mpFuncTest2.csv',
                                             burstData=sample,
                                             expEHist=exp_E_Hist,
                                             EBins=Ebins,
                                             E1min=E1min,
                                             E2min=E2min,
                                             K=5,
                                             N=100)
    # Dictionary function
    # test = runPDAOverkSpace_MultiProcess_DictListOutput(ks=kSpaceTotal,
    #                                                     PROCESSOR_COUNT=20,
    #                                                     burstData=sample,
    #                                                     expEHist=exp_E_Hist,
    #                                                     EBins=Ebins,
    #                                                     E1min=E1min,
    #                                                     E2min=E2min,
    #                                                     K=5,
    #                                                     N=100)
    
    # print(test)
