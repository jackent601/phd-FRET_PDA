import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import LinearNDInterpolator

def unpackDFValues(sseResultsDF, mean=False):
    X = np.array(sseResultsDF.k1)
    Y = np.array(sseResultsDF.kminus1)
    if mean:
        Z = np.array(sseResultsDF.SSE_mean)
    else:
        Z = np.array(sseResultsDF.SSE)
    return X, Y, Z

def printLowestSSEfromDF(sseResultsDF, SSEFeatureName='SSE'):
    lowestSSE = sseResultsDF.sort_values(SSEFeatureName).iloc[0, :]
    print(f'lowest SSE from {len(sseResultsDF)} samples')
    print(lowestSSE)
    
def getInterpolationfromDF(sseResultsDF, logSpace=False):
    # Get DF Values
    X, Y, Z = unpackDFValues(sseResultsDF)
    
    # Adjust for log
    if logSpace:
        X = np.log10(X)
        Y = np.log10(Y)
    
    # Get Interpolation (Required in both cases)
    return LinearNDInterpolator(list(zip(X,Y)), Z)
    
def get3DSSEFromResultsDF(sseResultsDF, savePlotPath=None, logSpace=False, title=None, show=True, mean=False):
    # Get DF Values
    X, Y, Z = unpackDFValues(sseResultsDF, mean=mean)
    
    # Adjust for log
    if logSpace:
        X = np.log10(X)
        Y = np.log10(Y)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, alpha=0.5, cmap = 'jet')
    ax.view_init(30, 200)
    ax.set_xlabel('k1')
    ax.set_ylabel('kminus1')
    if title is not None:
        plt.title(title)
    if savePlotPath is not None:
        plt.savefig(savePlotPath)
    if show:
        plt.show()
    
def getHeatPlotFromSSEDF(sseResultsDF, 
                         savePlotPath=None, 
                         logSpace=False,
                         interpolation='antialiased', 
                         title=None, show=True, mean=False):
    # Get DF Values
    X, Y, Z = unpackDFValues(sseResultsDF, mean=mean)
    
    # Adjust for log
    if logSpace:
        X = np.log10(X)
        Y = np.log10(Y)
    
    # Get Interpolation (Required in both cases)
    interp = LinearNDInterpolator(list(zip(X,Y)), Z)
    _X = np.unique(X)
    _Y = np.unique(Y)
    _Xgrido, _Ygrido = np.meshgrid(_X, _Y)  # 2D grid for interpolation
    _Zgrido = interp(_Xgrido, _Ygrido)
    
    # Plot
    plt.imshow(_Zgrido,
               extent=[min(X), max(X), min(Y), max(Y)],
               origin="lower",
               interpolation=interpolation,
               cmap='jet')
    plt.colorbar()
    if title is not None:
        plt.title(title)
    if savePlotPath is not None:
        plt.savefig(savePlotPath)
    if show:
        plt.show()
    
def getInterpolatedHeatPlotFromSSEDF(sseResultsDF, 
                                     interpolation,
                                     savePlotPath=None, 
                                     logSpace=False,
                                     title=None,
                                     show=True, mean=False):
    return getHeatPlotFromSSEDF(sseResultsDF,
                                savePlotPath=savePlotPath,
                                logSpace=logSpace,
                                interpolation=interpolation,
                                title=title,
                                show=show, mean=mean)