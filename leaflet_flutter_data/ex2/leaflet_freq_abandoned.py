
from mpl_toolkits.mplot3d import Axes3D
import time

import numpy as np
import scipy
from scipy import fft
import matplotlib.pyplot as plt
# import tensorflow as tf
# import tensorflow.compat.v2 as tf
# import tensorflow_probability as tfp
# tfb = tfp.bijectors
# tfd = tfp.distributions
# tfk = tfp.math.psd_kernels
# tf.enable_v2_behavior()

# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'



data_dir = '/home/chaztikov/git/aorta_piv_data/data/original/'
fname = 'OpenAreaPerimountWaterbpm60.txt'


data = np.loadtxt(data_dir+fname)

# A = textscan(fileID,'%f %f','Delimiter',',');

# %Edit BPM
BPM = 80;
InitialFrame =  000;
StartFFTFrame = 1;
NFFT = 512*4;
FPS = 5000;
PixelsPerCm = 100;


# Area = A{1,2}/(PixelsPerCm^2);
Area = data.copy(); #[0,1]
Area = Area[:,1]
Time =np.arange(1,FPS,1)/FPS*1000;
FramesPerPulse = int(60/BPM*FPS);
Pulses = np.floor(( Area.shape[0] - InitialFrame)/FramesPerPulse);

Pulses=int(Pulses)
PulseData=np.zeros([Pulses,FramesPerPulse])

PulseData = Area.copy()
PulseData.shape
PulseData+=InitialFrame
ncut=np.mod(PulseData.shape[0],FramesPerPulse)


PulseData=PulseData[:-ncut]
# PulseData = PulseData[:-np.mod(PulseData.shape[0],Pulses)]

# assert(np.mod(PulseData.shape[0],Pulses)==0)
# assert(np.mod(PulseData.shape[0],FramesPerPulse)==0)

PulseData = PulseData.reshape([Pulses,FramesPerPulse])

# for i in range(Pulses):
#     for j in range(FramesPerPulse):
#         PulseData[i,j] = Area[ (i-1)*FramesPerPulse + j + InitialFrame ]

plt.figure()
plt.title('Open Area vs. Frame')
plt.xlabel('Frame')
plt.ylabel('Valve Open Area [cm^2]')
plt.plot(PulseData.T,'.')
plt.show()


'''
FFT
'''
PulseFreq = FPS*( np.arange(0,NFFT//2))/NFFT;

# PulseDataFFT=np.zeros(PulseData.shape)
# fft(PulseData(i,StartFFTFrame:(StartFFTFrame+NFFT-1)))

PulseDatafft = fft(PulseData)
power = np.abs(PulseDatafft)/NFFT

plt.figure()
plt.plot(PulseFreq,power[:NFFT//2])
plt.title('FFT Open Area ')
plt.xlabel('Frequency')
plt.ylabel('FFT Valve Open Area [cm^2]')
plt.show()
print(power.argsort())


import scipy.signal
from scipy.signal import blackman
w = blackman(PulseData.shape[0])
PulseDatawfft = fft(PulseData * w)
wpower = np.abs(PulseDatawfft)/NFFT

plt.figure()
plt.plot(PulseFreq,
         power[:NFFT//2],
         '.')
plt.title('FFT Open Area ')
plt.xlabel('Frequency')
plt.ylabel('FFT Valve Open Area [cm^2]')
plt.show()

print(wpower.argsort(), '\n', wpower[wpower.argsort()])


# for i=1:Pulses
#     for j=1:FramesPerPulse
#         PulseData(i,j) = Area((i-1)*FramesPerPulse+j+InitialFrame);
#     end
#     plot(PulseData(i,:))
#     drawnow
# end



# %perform FFT
# PulseFreq = FPS*(0:(NFFT/2))/NFFT;
# figure(2)
# hold on
# title('Valve Area Frequency')
# xlabel('Frequency [Hz]')
# ylabel('|power|')
# set(gca,'FontSize',20)
# xlim([0 50])
# for i=1:Pulses
#     PulseDataFFT(i,:) = fft(PulseData(i,StartFFTFrame:(StartFFTFrame+NFFT-1)));
#     power = abs(PulseDataFFT(i,:)/NFFT);
#     plot(PulseFreq,power(1:NFFT/2+1))
# end
# ylim([0 1])


# %Find peak
# figure(3)
# hold on
# title('Fluttering Frequency (first peak)')
# xlabel('Frequency [Hz]')
# ylabel('Number of Pulses')
# set(gca,'FontSize',20)
# Peaks=zeros(size(PulseFreq,2),1);
# for i=1:Pulses
#     power = abs(PulseDataFFT(i,:)/NFFT);
#     j=1;
#     while power(j)-power(j+1)>0 || power(j+2)-power(j+1)>0
#         j=j+1;
#     end
#     Peaks(j+1) = Peaks(j+1)+1;
# end
# bar(PulseFreq,Peaks)
# xlim([0 30])

# filename = sprintf('FlutteringFrequency%dbpm.png',BPM);
# saveas(gcf,filename)

# %difference from smoothed
# figure(4)
# hold on
# title('Fluttering Frequency (Difference from smoothed)')
# xlabel('Frequency [Hz]')
# ylabel('Number of Pulses')
# set(gca,'FontSize',20)
# Peaks=zeros(size(PulseFreq,2),1);
# for i=1:Pulses
#     PulseDataSmoothed = smooth(PulseData(i,:),150);
#     Variance = PulseData(i,:)-PulseDataSmoothed';
#     VarianceFFT(i,:) = fft(Variance(StartFFTFrame:(StartFFTFrame+NFFT-1)));
#     power = abs(VarianceFFT(i,:)/NFFT);
#     plot(PulseFreq,power(1:NFFT/2+1))
# end
# xlim([0 50])
# filename = sprintf('VarianceFrequency%dbpm.png',BPM);
# saveas(gcf,filename)