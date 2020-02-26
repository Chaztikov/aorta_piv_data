
import os
import sys
import re
import numpy as np
import pandas as pd
import subprocess
import pandas as pd

import numpy as np
import numpy.fft as fft
from numpy.fft import fftfreq


import scipy
import scipy.integrate


from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot, make_interp_spline
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os
import sys
import re
import numpy as np
import pandas as pd
import subprocess
import pandas as pd
import numpy as np
import scipy
import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot, make_interp_spline
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

# import seaborn as sns
import sklearn.decomposition
from sklearn.decomposition import PCA
from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
dimreductiontype = 'pca'

interpolate_signal = 1

cwd = os.getcwd()
# dname=cwd+'/save/'
# dname=cwd+'/'



dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
fnames = os.listdir(dname)
fnames0 = 'OpenAreaPerimountWaterbpm'
fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]

bpmdatas = np.array(
    np.loadtxt(dname+'bpmdata.txt', unpack=False)
    , dtype=int)


datalist=[]
for fname0 in fnames[:]:
    print(fname0,'\n\n')
    data = np.loadtxt(dname+fname0)[:, :]
    datalist.append(data)

minlen=min([len(dat) for dat in datalist])
datalist=[dat[:minlen] for dat in datalist]
datalist=np.array(datalist)
times=datalist[:,:,0]
datalist=datalist[:,:,1]
# plt.figure();plt.imshow(datalist.T);plt.show()
inz=[np.nonzero(dat)[0] for dat in datalist]
nzdatalist = [dat[inz[idat]] for idat,dat in enumerate(datalist)]
[len(dat) for dat in nzdatalist]
nzdatalist = np.stack(nzdatalist)

nzdata=[np.nonzero(dat)[0] for dat in datalist];nzdatalens=[len(dat) for dat in nzdata];minlen=min(nzdatalens);maxlen=max(nzdatalens)
nzdatalist=([datalist[ii,nzdat[:minlen]][:] for ii,nzdat in enumerate(nzdata)])
nzdatalist=np.stack(nzdatalist)


nzdata
dat[:min]
plt.plot(nzdatalist.T);plt.show()
import scipy.signal
import scipy.interpolate as interp

interp.splint(times[:itf],nzdatalist[0,:itf])

# xx=np.linspace(times[0,0],times[0,itf],2*len(times[0,:itf]))
# spl
# spl=
# scipy.fit
# help(scipy.signal.bspline)
# scipy.signal.bspline(xx,3)
# spl(xx)
# spl(xx)

itf=np.where(np.diff(nzdata[0])>1)[0][0]
nzdatalist

diff2=np.diff((np.diff((datalist>0),axis=1).astype(float)))
adiff2=np.abs(diff2)
plt.plot(adiff2[0]);plt.show()
nzad2 = [np.where(ad2>0)[0][:] for ad2 in adiff2]
# nzad2[0]

# yy = datalist[0]
# yy.shape
# yhat,pxx_den = scipy.signal.periodogram(yy)

# N=yy.shape[0]
# fs=(times.shape[0]/times.max())**(-1)
# scipy.signal.spectrogram(yy, fs=fs, window=('tukey', 0.25), nperseg=None, noverlap=None, nfft=None, detrend='constant', return_onesided=True, scaling='density', axis=-1, mode='psd')
# plt.plot(yhat,pxx_den)
# plt.show()
# np.argmax(pxx_den)
# pxx_den[13]
# yhat[13]
# np.sqrt(pxx_den.max())




itf=np.where((diff2>0)[0])[0][0]
it0 = itf + 1
itf = it0 + itf
xx=np.linspace(times[0,0],times[0,itf],2*len(times[0,:itf]))
x=times[0,it0:itf]
y=nzdatalist[0,it0:itf]
x=x[:y.shape[0]]
(x==0).sum()
(y==0).sum()
x+x[1]
x.shape[0]
y.shape
itf
f=np.linspace(x[0],x[-1],x.shape[0]* 1 )
pgram = scipy.signal.lombscargle(x, y, f, normalize=True)

print(f[np.argmax(y)]*np.sqrt(2))
print(f[np.argmax(y)])
1/.00925
(x-(f[itf]-f[it0]))

print(0.5 / f[np.argmax(y)])
plt.figure()
plt.subplot(2,1,1)
plt.plot(x,y,'b+')
plt.subplot(2,1,2)
plt.plot(f,pgram)
plt.grid()
# plt.xlim([120,130])
plt.savefig('test')
plt.show()

#freq. at peak
# >>> nzdatalist[0,:itf].shape #1 cycle;
# >>> nzdatalist[0,::itf].shape #num_cycles
plt.plot(nzdatalist[0,:itf-it0+1],'o');plt.show()

itau=itf-it0;

samples=np.array([[row] for row in nzdatalist[0,::itau]])
samples.shape

ntau = int(np.floor(nzdatalist[0].shape[0]/itau))
ntau
nzdatalist[0].shape[0]/itau
sdata=datalist[0][itau:ntau*itau].reshape([itau,ntau])
plt.plot(sdata[-itau//3:,-1:]);plt.show()
sdata.shape
ntau


for fname0 in fnames[:1]:
    print(fname0,'\n\n')
    data = np.loadtxt(dname+fname0)[:, :]


    x,y = data[:,0],data[:,1]
    inz = np.nonzero(y)[0]
    x=x[inz]
    y=y[inz]
    
    y=np.sin(x)
    
    nx=x.shape[0]
    dx=np.diff(x)[0]
    
    dt = (x[-1]-x[0])//dx
    dt = int(dt)

    fy = np.fft.fft(y)
    fy = np.abs(fy[:nx]) / nx
    
    # freqs=np.fft.fftfreq(N,dt)
    # freqs = np.arange(-nx//2, nx//2, 1) // 2 / float(dt*N)
    freqs = np.arange(-nx//2, nx//2) / nx
    
    print(freqs)
    print(fy.shape, freqs.shape, nx, y.shape)

    plt.figure(figsize=(24, 12))
    plt.plot(freqs, fy, '-o')
    plt.title(fname0)
    plt.grid()
    plt.xlabel('f')
    plt.ylabel('|F(f)|')
    plt.savefig('freqsamples_'+fname0[-6:-4]+'.png')
    plt.show()

    period = int(1//freqs[::-1][0])//nx
    period