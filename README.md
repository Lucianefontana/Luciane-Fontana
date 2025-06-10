# Time series analysis # 

Basics Fourier Analysis 

---


4. Fourier Analysis.ipynb

Data For Science, Inc
Time Series Analysis

Fourier Analysis


import datetime

import numpy as np
import pandas as pd

import scipy as sp
import scipy.fftpack

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import watermark

%matplotlib inline
%load_ext watermark
List out the versions of all loaded libraries

%watermark -n -v -m -g -iv
Python implementation: CPython
Python version       : 3.8.5
IPython version      : 7.19.0

Compiler    : Clang 10.0.0 
OS          : Darwin
Release     : 22.3.0
Machine     : x86_64
Processor   : i386
CPU cores   : 16
Architecture: 64bit

Git hash: d1697f4b6ce27d8e1a59727a1d7b4bf47b92104d

matplotlib: 3.3.2
watermark : 2.1.0
pandas    : 1.5.3
scipy     : 1.6.1
json      : 2.0.9
numpy     : 1.24.2

Set the default figure style

plt.style.use('./d4sci.mplstyle')
Generate some fake data
t_max = 4 # signal duration (seconds)
sample_freq = 250 # points per second
N = t_max*sample_freq
t = np.linspace(0, t_max, N)
amp = np.array([1, .3, .1])
freq = np.array([1, 2, 10])
Plot the individual components and the total signal

total = np.zeros(N)
components = []

n_freq = len(freq)

for i in range(n_freq):
    current = amp[i]*np.cos(2*np.pi*freq[i]*t)
    total += current
    
    components.append(current)
    plt.plot(t, current, label='f='+str(freq[i]), lw=1)

    
plt.plot(t, total, label='total')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
<matplotlib.legend.Legend at 0x7fa56077db50>

Get the color cycle

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
Vizualize in 3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel("\n\nTime (s)", fontsize=18)
ax.set_ylabel("\n\nFrequency (Hz)", fontsize=18)
ax.set_zlabel("\n\nAmplitude", fontsize=18)

linewidth = 1

n_freq = np.max(freq)+2
x = np.linspace(0,4,1000)
y = np.ones(x.size)

# Plot the total signal
ax.plot(x, y*n_freq, total, linewidth=3, color=colors[1])

# Plot the amplitudes
z = np.zeros(n_freq*100)
z[freq*100] = amp

ax.plot(np.zeros(n_freq*100), np.linspace(0, n_freq, n_freq*100), z, 
        linewidth=3, color=colors[3])

# Plot the components
y = np.ones(1000)
for i in range(0, len(components)):
    ax.plot(x, y*freq[i], components[i], linewidth=1.5, color=colors[0])
    
ax.set_yticks(freq)
ax.set_yticklabels(freq)

ax.set_xlim(0, t_max)
ax.set_ylim(0, n_freq)
(0.0, 12.0)
/Users/bgoncalves/opt/anaconda3/lib/python3.8/site-packages/IPython/core/pylabtools.py:132: UserWarning: Tight layout not applied. The left and right margins cannot be made large enough to accommodate all axes decorations. 
  fig.canvas.print_figure(bytes_io, **kw)

We can also recover the original frequencies and amplitudes from the total signal, by taking the fourier transform

fft_values = scipy.fftpack.fft(total)
The result is an array of complex numbers

fft_values.dtype
dtype('complex128')
To recover the real component of the signal, we simply take the absolute value. The imaginary components corresponds to phase information

fft_real = 2.0/N * np.abs(fft_values[0:N//2])
We see that only a few values are significantly different from zero:

np.where(fft_real>0.01)
(array([ 4,  8, 40]),)
To properly recover the corresponding freqency values, we must calculate the freqency resolution

freq_resolution = sample_freq/N
This is the value we need to convert indices into frequencies

freq_values = np.arange(N)*freq_resolution
So now we can plot the recovered amplitudes with the matching frequencies. For clarity, we plot only the first 50 values and the compare with the original input values

plt.plot(freq_values[:50], fft_real[:50], label='calculated')
plt.scatter(freq, amp, s=100, color=colors[1], zorder=3, label='original')
plt.xlabel('Freq (Hz)')
plt.ylabel('Amplitude')
plt.legend()
<matplotlib.legend.Legend at 0x7fa550a139a0>

Load ILI dataset
ILI = pd.read_csv('data/CDC.csv')
ILI['date'] = ILI['Year']+ILI['Week']/52.
Visualize it

ILI.plot(x='date', y=['Percent of Deaths Due to Pneumonia and Influenza', 'Expected', 'Threshold'])
ax = plt.gca()
ax.legend(['Mortality', 'Expected', 'Threshold'])
ax.set_xlabel('Date')
ax.set_ylabel('% Mortality')
Text(0, 0.5, '% Mortality')

Calculate the FFT

signal = ILI['Percent of Deaths Due to Pneumonia and Influenza']
date = ILI['date']
N = len(signal)
fft_values = scipy.fftpack.fft(signal.values)
And the frequencies

freq_values = scipy.fftpack.fftfreq(N, 1/52) # 52 weeks per year
Plot the amplitude as a function of frequency

fig, ax = plt.subplots(1)
ax.semilogy(freq_values[:N//2], np.abs(fft_values[:N//2]))
ax.set_xlim(0, 5)
ax.set_xlabel('Frequency (1/year)')
ax.set_ylabel('Amplitude')
Text(0, 0.5, 'Amplitude')

The frequency 0 component corresponds to a constant "level" after which the strongest component is ~1 year, indicating the yearly seasonality we had already identified. Higher freqencies have increasingly smaller Amplitudes indicating their decreasing importance.

We can remove some of the noise in the data by filtering out some of the higher frequencies. If we set every frequency higher than 2/year to zero

filtered = fft_values.copy()
filtered[np.abs(freq_values) > 2] = 0
And reconstrucing the original dataset using the filtered frequencies

signal_filtered = np.real(sp.fftpack.ifft(filtered))
We obtain a cleaner version of the signal

fig, ax = plt.subplots(1)
ax.plot(date, signal, lw=1, label='original')
ax.plot(date, signal_filtered, label=r'$f_{max}=2$')
ax.set_xlabel('Date')
ax.set_ylabel('% Mortality')
fig.legend()
<matplotlib.legend.Legend at 0x7fa590e4f250>

Naturally, the more frequencies we include the closer we get to the original dataset

filtered2 = fft_values.copy()
filtered2[np.abs(freq_values) > 4] = 0
signal_filtered2 = np.real(sp.fftpack.ifft(filtered2))
fig, ax = plt.subplots(1)
ax.plot(date, signal, lw=1, label='original')
ax.plot(date, signal_filtered, lw=1, label=r'$f_{max}=2$')
ax.plot(date, signal_filtered2, label=r'$f_{max}=4$')
ax.set_xlabel('Date')
ax.set_ylabel('% Mortality')
plt.legend()
<matplotlib.legend.Legend at 0x7fa540939610>

Data For Science, Inc
