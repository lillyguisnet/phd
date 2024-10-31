import pandas as pd

df = pd.read_csv('/home/lilly/phd/wellswimming/df_swim.csv')

df.head()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.fft import fft

# Assuming your data is in a pandas DataFrame called 'df'
# with a column 'bends' for the number of bends and 'time' for the time points

# 1. Autocorrelation
def plot_acf(data, lags=50):
    acf_values = acf(data, nlags=lags)
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(acf_values)), acf_values)
    plt.title('Autocorrelation Function')
    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.show()
    plt.savefig("plot_acf.png")
    plt.close()

# 2. Fourier Transform
def plot_fft(data, sampling_rate):
    n = len(data)
    yf = fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*sampling_rate), n//2)
    plt.figure(figsize=(10, 5))
    plt.plot(xf, 2.0/n * np.abs(yf[0:n//2]))
    plt.title('FFT of Time Series')
    plt.xlabel('Frequency')
    plt.ylabel('Magnitude')
    plt.show()
    plt.savefig("plot_fft.png")
    plt.close()

# Example usage:
# plot_acf(df['bends'])
# plot_fft(df['bends'], 1/np.mean(np.diff(df['minutes'])))

# 3. Periodogram
from scipy import signal

def plot_periodogram(data, fs):
    f, Pxx = signal.periodogram(data, fs)
    plt.figure(figsize=(10, 5))
    plt.semilogy(f, Pxx)
    plt.title('Periodogram')
    plt.xlabel('Frequency')
    plt.ylabel('Power Spectral Density')
    plt.show()
    plt.savefig("plot_periodogram.png")
    plt.close()

# Example usage:
# plot_periodogram(df['bends'], 1/np.mean(np.diff(df['minutes'])))

# 4. Wavelet Analysis
import pywt

def plot_wavelet(data, scales, wavelet='cmor'):
    coef, freqs = pywt.cwt(data, scales, wavelet)
    plt.figure(figsize=(10, 5))
    plt.imshow(np.abs(coef), extent=[1, len(data), 1, len(scales)], 
               aspect='auto', interpolation='nearest')
    plt.title('Wavelet Transform')
    plt.ylabel('Scale')
    plt.xlabel('Time')
    plt.colorbar(label='Magnitude')
    plt.show()
    plt.savefig("plot_wavelet.png")
    plt.close()

# Example usage:
# scales = np.arange(1, 128)
# plot_wavelet(df['bends'], scales)



### Split by condition ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.fft import fft
from scipy import signal
import pywt
import os

def plot_grouped_analysis(df, group_col, value_col, analysis_func):
    groups = df[group_col].unique()
    
    fig, axes = plt.subplots(len(groups), 1, figsize=(10, 5*len(groups)), squeeze=False)
    
    for i, group in enumerate(groups):
        subset = df[df[group_col] == group]
        if not subset.empty:
            ax = axes[i, 0]
            analysis_func(subset[value_col], ax)
            ax.set_title(f"{group}, Condition: 0.0")
    
    plt.tight_layout()
    return fig

def plot_acf(data, ax, lags=50):
    acf_values = acf(data, nlags=lags)
    ax.bar(range(len(acf_values)), acf_values)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')

def plot_fft(data, ax):
    n = len(data)
    yf = fft(data)
    xf = np.linspace(0.0, 1.0/(2.0*5), n//2)  # Assuming 1 minute intervals
    ax.plot(xf, 2.0/n * np.abs(yf[0:n//2]))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Magnitude')

def plot_periodogram(data, ax):
    f, Pxx = signal.periodogram(data, 5)  # Assuming 1 minute intervals
    ax.semilogy(f, Pxx)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('PSD')

def plot_wavelet(data, ax, scales=np.arange(1, 128)):
    coef, freqs = pywt.cwt(data, scales, 'cmor')
    ax.imshow(np.abs(coef), extent=[1, len(data), 1, len(scales)], 
              aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_ylabel('Scale')
    ax.set_xlabel('Time')

def analyze_grouped_periodicity(df, group_col='condition', condition_col='mc_concentration', value_col='bends'):
    # Filter the dataframe to include only rows where condition is "0.0"
    df_filtered = df[df[condition_col] == 0.0]
    
    analyses = [
        ('Autocorrelation_Function', plot_acf),
        ('FFT_of_Time_Series', plot_fft),
        ('Periodogram', plot_periodogram),
        ('Wavelet_Transform', plot_wavelet)
    ]
    
    os.makedirs('periodicity_plots', exist_ok=True)
    
    for title, func in analyses:
        print(f"\nGenerating {title} plots for condition 0.0...")
        fig = plot_grouped_analysis(df_filtered, group_col, value_col, func)
        
        filename = os.path.join('periodicity_plots', f'{title}_condition_0.0.png')
        fig.savefig(filename)
        plt.close(fig)
        print(f"Saved {filename}")

# Usage
analyze_grouped_periodicity(df, group_col='condition', condition_col='mc_concentration', value_col='bends')




# New code block for autocorrelation plot containing all conditions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import os

# Load the data (assuming df is already loaded)
# df = pd.read_csv('/home/lilly/phd/wellswimming/df_swim.csv')

def plot_acf_all_conditions(df, group_col='condition', value_col='bends', lags=50):
    conditions = sorted(df[group_col].unique())
    viscosities = sorted(df['mc_concentration'].unique())
    
    fig, axes = plt.subplots(len(conditions), len(viscosities), figsize=(20, 5*len(conditions)), squeeze=False)
    
    condition_titles = {
        'a': 'Agar ancestry, Agar growth',
        'b': 'Agar ancestry, Scaffold growth',
        'c': 'Scaffold ancestry, Scaffold growth',
        'd': 'Scaffold ancestry, Agar growth'
    }
    
    for i, condition in enumerate(conditions):
        for j, viscosity in enumerate(viscosities):
            subset = df[(df[group_col] == condition) & (df['mc_concentration'] == viscosity)]
            if not subset.empty:
                ax = axes[i, j]
                acf_values = acf(subset[value_col], nlags=lags)
                ax.bar(range(len(acf_values)), acf_values)
                ax.set_xlabel('Lag')
                ax.set_ylabel('ACF')
                ax.set_title(f"{condition_titles[condition]}, Viscosity (MC %): {viscosity}")
    
    plt.tight_layout()
    return fig


fig = plot_acf_all_conditions(df)

filename = os.path.join('wellswimming/Autocorrelation_Function_all_conditions_paper.png')
fig.savefig(filename)
plt.close(fig)


# New code block for wavelet transform plot containing all conditions
def plot_wavelet_all_conditions(df, group_col='condition', value_col='bends', scales=np.arange(1, 128), wavelet='cmor'):
    conditions = sorted(df[group_col].unique())
    viscosities = sorted(df['mc_concentration'].unique())
    
    fig, axes = plt.subplots(len(conditions), len(viscosities), figsize=(20, 5*len(conditions)), squeeze=False)
    
    condition_titles = {
        'a': 'Agar ancestry, Agar growth',
        'b': 'Agar ancestry, Scaffold growth',
        'c': 'Scaffold ancestry, Scaffold growth',
        'd': 'Scaffold ancestry, Agar growth'
    }
    
    # Calculate global magnitude range
    all_coefs = []
    for condition in conditions:
        for viscosity in viscosities:
            subset = df[(df[group_col] == condition) & (df['mc_concentration'] == viscosity)]
            if not subset.empty:
                coef, _ = pywt.cwt(subset[value_col], scales, wavelet)
                all_coefs.append(np.abs(coef))
    
    vmin = min(np.min(coef) for coef in all_coefs)
    vmax = max(np.max(coef) for coef in all_coefs)
    
    for i, condition in enumerate(conditions):
        for j, viscosity in enumerate(viscosities):
            subset = df[(df[group_col] == condition) & (df['mc_concentration'] == viscosity)]
            if not subset.empty:
                ax = axes[i, j]
                coef, freqs = pywt.cwt(subset[value_col], scales, wavelet)
                im = ax.imshow(np.abs(coef), extent=[1, len(subset), 1, len(scales)], 
                               aspect='auto', interpolation='nearest', cmap='viridis',
                               vmin=vmin, vmax=vmax)
                ax.set_ylabel('Scale')
                ax.set_xlabel('Time')
                ax.set_title(f"{condition_titles[condition]}, Viscosity (MC %): {viscosity}")
                
                # Add colorbar
                plt.colorbar(im, ax=ax, label='Magnitude')
    
    plt.tight_layout()
    return fig

# Generate the wavelet transform plot
fig_wavelet = plot_wavelet_all_conditions(df)

# Save the figure
filename_wavelet = os.path.join('wellswimming', 'Wavelet_Transform_all_conditions_paper.png')
fig_wavelet.savefig(filename_wavelet)
plt.close(fig_wavelet)
