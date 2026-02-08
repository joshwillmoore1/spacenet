import matplotlib.pyplot as plt 
import numpy as np

def plot_correlation_function(radii,pcf_values,confidence_interval=None,ax=None):
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8,6))
    
    if pcf_values.shape[0] == 1:
        ax.plot(radii, pcf_values, label='Pair Correlation Function', color='tab:blue')
        if confidence_interval is not None:
            lower_bound, upper_bound = confidence_interval
            ax.fill_between(radii, lower_bound, upper_bound, color='tab:blue', alpha=0.5, label='Confidence Interval')
        
    elif pcf_values.shape[0] == 2:
        raise NotImplementedError("wpcf plotting not implemented yet.")

    
    ax.legend()
    ax.grid(True)
    
    return plt.gcf(), ax