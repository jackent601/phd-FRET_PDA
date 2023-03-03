import argparse
import json
import pandas as pd

import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# from scipy.optimize import minimize
import os

import sys
sys.path.append('./../')
import modules.modulesCorrectionFactorsAndPlots as MCF
import modules.modulesPopulationFitting as PF
import PDA as PDA
import PDA_SSEResults as SSE
# import TwoStateKineticModel as KM2S
# import PDA_FastDescent as PDA_FD

# See docs for full description of each method
OPTIMISATION_METHODS = ['EndToEnd', 
                        'LogSearch', 
                        'Binned', 
                        'Burst']
LOG_SEARCH_METHODS = ['EndToEnd']

if __name__ == '__main__':
    # Define Parser for CLU
    parser = argparse.ArgumentParser(description='Probability Distribution Analysis for two state kinetic model, including gaussian spreading')
    parser.add_argument('-c','--config_path', help='Path to config file for parameter optimisation, see documentation for details', required=True)
    parser.add_argument("-v", "--verbose", help="increase output verbosity",action="store_true")
    
    # Get Arguments
    args = parser.parse_args()
    arg_vars = vars(args)
    
    # Load Config file with optimisation details
    with open(arg_vars['config_path'], "r") as config_file:
        CONFIG = json.load(config_file)
        
    # Debug
    if args.verbose:
        print(CONFIG)
    
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # Unpack & Check Config
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    DATA_PATH = CONFIG['DATA_PATH']
    ROOT_DIR = CONFIG['ROOT_DIR']
    FRET2CDEmask = CONFIG['FRET2CDEmask']
    GroupFeatureName = CONFIG['GroupFeatureName']
    EBinLower = CONFIG['EBinning']['EBinLower']
    EBinUpper = CONFIG['EBinning']['EBinUpper']
    NEBins = CONFIG['EBinning']['NEBins']
    OPTIMISATION_METHOD = CONFIG['OPTIMISATION_METHOD']
    N0 = CONFIG['N']
    
    # GAUSSIAN_PARAMETERS
    r0 = CONFIG['GAUSSIAN_PARAMETERS']['r0']
    rDeviation = CONFIG['GAUSSIAN_PARAMETERS']['rDeviation']
    gaussianResolution = CONFIG['GAUSSIAN_PARAMETERS']['gaussianResolution']
    
    # Check optimisation method is handled
    assert OPTIMISATION_METHOD in OPTIMISATION_METHODS, "optimisation method not handled, see documentation for choices and descriptions"
    
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # Read Data
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    burst_data = pd.read_csv(DATA_PATH)
    # Filter out A/D - only populations, keep high FRET 2CDE as dont want to filter on this
    burst_data = MCF.typical_S_ALEX_FRET_CDE_filter(burst_data, FRET2CDEmask = FRET2CDEmask)
    # Get unique groups
    groups = sorted(set(burst_data[GroupFeatureName]))
    
    # Debug
    if args.verbose:
        print(f'groups: {groups}')

    # E bins
    Ebins, EBinCentres, dBin = PF.getBins(EBinLower,EBinUpper,NEBins)
    
    if args.verbose:
        print(Ebins)
        
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # Log Searching
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    if OPTIMISATION_METHOD in LOG_SEARCH_METHODS:
        logkRange = np.linspace(CONFIG['logkRange']['Lower'], CONFIG['logkRange']['Upper'], CONFIG['logkRange']['Sep'])
        kRange = [10**i for i in logkRange]
        if args.verbose:
            print(f'k values:')
            for k in kRange:
                print(f'\t {k:0.2f}')
        
        # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
        # loop through groups & Run optimisation
        # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
        logPrefix = 'logkSearch'
        logDir = os.path.join(ROOT_DIR)
        os.makedirs(ROOT_DIR, exists_ok=True)
        os.makedirs(os.path.join(ROOT_DIR), exists_ok=True)
        for g in groups:
            g_subset = burst_data[burst_data[GroupFeatureName] == g]
            
            # Get Experimental E histogram
            exp_E = g_subset['E3_avg_beta_gamma']
            exp_E_Hist, _ = np.histogram(exp_E, Ebins)

            # Estimate good starting candidates for E1 and E2 by filtering out dynamics with FRET2CDE
            Eestimation = MCF.typical_S_ALEX_FRET_CDE_filter(g_subset)
            b, g, expEs, expSs = MCF.calculate_beta_and_gamma_correction(None,
                                                                        population_E_limits=[ [0, 0.5], [0.501, 1]],
                                                                        E=Eestimation['E3_avg_beta_gamma'],
                                                                        S=Eestimation['S3_avg_beta_gamma'],
                                                                        sample_percent=0.05,
                                                                        plot_fig=False)
            _E1_0, _E2_0 = expEs
            
            if args.verbose:
                print(f'Group: {g}, E1_0: {_E1_0:0.2f}, E2_0: {_E2_0:0.2f}')
                
            
            # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
            # Run log k-space search
            # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
            # Full Method, run log search to find initial guesses of k1, k-1 and check loss function will be smooth
            _savepath = os.path.join(logDir, f'{logPrefix}_{g}')
            logResults = []
            for _k1 in kRange:
                for _kminus1 in kRange:
                    # Run PDA (gaussian) over log k-space
                    pEsn_ForkPair = PDA.get_pEsn_fromBurstDataFrame_withGaussian(BurstData=g_subset,
                                                                                Ebins=Ebins,
                                                                                EbinCentres=EBinCentres,
                                                                                E1=_E1_0,
                                                                                E2=_E2_0,
                                                                                N=N0,
                                                                                k1=_k1,
                                                                                kminus1=_kminus1,
                                                                                r0=r0,
                                                                                rDeviation=rDeviation,
                                                                                gaussianResolution=gaussianResolution)
                    # Get loss function
                    sse_ForkPair = np.sum(np.array(np.array(exp_E_Hist) - np.array(pEsn_ForkPair))**2)
                    # Add to results
                    logResults.append({'k1': _k1, 'kminus1': _kminus1, 'SSE':sse_ForkPair , 'pEsn':pEsn_ForkPair})
                    
            # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
            # Save & Plot Results
            # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
            # Save results as df
            logResults_df = pd.DataFrame(logResults)
            logResults_df.to_csv(f'{_savepath}.csv', index=False)

            # Plot loss function heatmap
            SSE.getHeatPlotFromSSEDF(logResults_df,
                                    logSpace=True,
                                    title=f'SSE heatplot, g: {g}',
                                    savePlotPath=f'{_savepath}_SSE_Heatmap.png',
                                    show=False)

            # Plot 3D loss function
            SSE.get3DSSEFromResultsDF(logResults_df,
                                    logSpace=True,
                                    title=f'3D SSE, g: {g}',
                                    savePlotPath=f'{_savepath}_SSE_3D_Heatmap.png',
                                    show=False)
        
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = 
    # Run Gradient Descent
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
