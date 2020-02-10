
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
dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
fnames = os.listdir(dname)
fnames0 = 'OpenAreaPerimountWaterbpm'
fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]

# bpm,iframe,ifframe=np.loadtxt('bpmdata.txt',unpack=True)
bpmdatas = np.loadtxt(dname+'bpmdata.txt', unpack=False)
bpmdatas = np.array(bpmdatas, dtype=int)
print(bpmdatas)

# for fname0 in fnames:
fname0 = 'OpenAreaPerimountWaterbpm100.txt'
# for fname0[-6:-4],fname0 in enumerate(fnames[:-2]):
# for fname0[-6:-4], fname0 in enumerate(fnames):

# data = np.loadtxt('datamatrix_'+fname0)[:, :]

data = np.loadtxt(dname+fname0)[:, :]

x,y = data[:,0],data[:,1]
ii0=20000
x=x[ii0:]
y=y[ii0:]
inz=y.nonzero()[0]
xx=x[inz]
yy=y[inz]
plt.plot(xx,yy);plt.show()
# plt.plot(xx,yy);plt.show()
icycle=yy.shape[0]//19-2
# np.mod(y.shape[0],19)/2
vv=[yy[ i*icycle: (i+1)*icycle] for i in range(yy.shape[0]//icycle)]
vmat=np.vstack(vv)[:-10]
plt.plot()
plt.plot(vmat.T)
plt.show()
np.savetxt('datamatrix_'+fname0,vmat)


