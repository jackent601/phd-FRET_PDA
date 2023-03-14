import argparse
import json
import pandas as pd
import sys
import os

from KineticFittingClass import *

sys.path.append('./../')
import FRET_modules.modulesPopulationFitting as PF

# import numpy as np
# # import matplotlib.pyplot as plt
# # import pandas as pd
# # from scipy.optimize import minimize
# import os

# import sys
# sys.path.append('./../')
# import modules.modulesCorrectionFactorsAndPlots as MCF
# import modules.modulesPopulationFitting as PF
# import PDA as PDA
# import PDA_SSEResults as SSE
# import TwoStateKineticModel as KM2S
# import PDA_FastDescent as PDA_FD

# See docs for full description of each method
OPTIMISATION_METHODS = ['EndToEnd', 
                        'LogSearch', 
                        'Binned', 
                        'Burst']
# LOG_SEARCH_METHODS = ['EndToEnd']

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
    GroupFeatureName = CONFIG['GroupFeatureName']

    EBinLower = CONFIG['EBinning']['EBinLower']
    EBinUpper = CONFIG['EBinning']['EBinUpper']
    NEBins = CONFIG['EBinning']['NEBins']
    EBins, EBinCentres, _ = PF.getBins(EBinLower,EBinUpper,NEBins)

    N_BURST_BINS = CONFIG['N_BURST_BINS']

    OPTIMISATION_METHOD = CONFIG['OPTIMISATION_METHOD']
    OPT_OPTIONS = CONFIG['OPT_OPTIONS']
    N0 = CONFIG['N']

    # GAUSSIAN_PARAMETERS
    r0 = CONFIG['GAUSSIAN_PARAMETERS']['r0']
    rDeviation = CONFIG['GAUSSIAN_PARAMETERS']['rDeviation']
    gaussianResolution = CONFIG['GAUSSIAN_PARAMETERS']['gaussianResolution']

    # Check optimisation method is handled
    assert OPTIMISATION_METHOD in OPTIMISATION_METHODS, "optimisation method not handled, see documentation for choices and descriptions"

    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    # Read & Optimise Data
    # = = = = = = = = = = = = = = = = = = = =  = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
    burst_data = pd.read_csv(DATA_PATH)
    # Group data
    burst_data_grouped = burst_data.groupby(GroupFeatureName)

    # Initialise Kinetic Fit Objects
    kineticFitDir = os.path.join(ROOT_DIR, 'fitObjects')
    os.makedirs(kineticFitDir, exist_ok=True)
    kfs = []
    for name, group in burst_data_grouped:
        kf = kineticFitting(rawBurstData=group, EBins=EBins, EBinCentres=EBinCentres, metaData={
                            'group_name': GroupFeatureName, 'group_value': name})
        kfs.append(kf)

        # Run various optmisations
        for i, kpair in enumerate([[10, 10], [100, 100], [1000, 1000], [10000, 10000]]):

            kf.optimiseBinnedPDA(saveDir=os.path.join(ROOT_DIR, f'{GroupFeatureName}_{name}/initial_pair_{i}'),
                                 intial_ks=kpair,
                                 nBurstBins=N_BURST_BINS,
                                 optimiseMethod='L-BFGS-B',
                                 optimiseOptions=OPT_OPTIONS,
                                 kbounds=((5, 5000), (5, 5000)))
        
        # Save kinetic fit object
        with open(os.path.join(kineticFitDir,f'kfObj_{GroupFeatureName}_{name}.pkl'), "wb") as f:
            pickle.dump(kf, f)
