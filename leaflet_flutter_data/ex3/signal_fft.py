
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