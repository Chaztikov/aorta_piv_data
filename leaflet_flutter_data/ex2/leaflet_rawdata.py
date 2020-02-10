
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

initframe=[12000,0,20000,21000]
nmaxima=[13,17,19,23]
splorder=1
for idx, fname0 in enumerate(fnames):

    data = np.loadtxt(dname+fname0)[:, :]

    x,y = data[:,0],data[:,1]

    ii0=initframe[idx]
    npeaks = nmaxima[idx]

    # plt.figure(figsize=(24,12))
    # plt.plot(y[ii0:])
    # plt.show()

    y = y[ii0:]
    x = x[ii0:]
    x0=x[0]
    ymax=y.max()
    y /=ymax
    # x-=x[0]
    
    # inz=y.nonzero()
    # x=x[inz]
    # y=y[inz]
    # y /=y.max()
    # x-=x[0]
    
    # plt.plot(xx,yy);plt.show()
    interp=BSpline(x,y,splorder)
    nx=x.shape[0]
    icycle = y.shape[0]//npeaks 
    icyclef = y.shape[0]/npeaks 
    print(icycle,'\n',icyclef)
    nx2 = nx*10
    
    # nx3 = int(nx*icycle / np.gcd(nx, icycle))
    # print(nx2,nx3)
    # nx2= nx3
    
    xx=np.linspace(x[0],x[-1], nx2)
    yy=interp(xx)
    inz2=yy.nonzero()
    xx=xx[inz2]
    yy=yy[inz2]
    # yy /=yy.max()
    # xx-=xx[0]
    
    icyclef = np.floor(yy.shape[0]/npeaks).astype(int)
    icycle = yy.shape[0]//npeaks 
    
    print('icyclef',icyclef)
    print('icycle',icycle)
    
    dicycle = np.mod(yy.shape[0],npeaks)
    print('dicycle',dicycle)
    # dicycle = 

    # icycle = icycle+dicycle//2
    # plt.figure(figsize=(24,12))
    # ncycles=4
    # plt.plot(xx[:icycle*ncycles],yy[:icycle*ncycles],'.');plt.show()a
    ny=yy.shape[0]//icycle
    # icycle=icycle
    ny
    # uu=[
    #     np.mod(xx[(i)*icycle:(i+1)*icycle], 
    #            xx[icycle])
    # uu=[
    #     np.sort(np.mod(xx[(i)*icycle:(i+1)*icycle], 
    #            xx[icycle]))
        #  for i in range(1,ny)]
    # yy*=ymax;
    # x+=x[0];
    uu=[xx[(0)*icycle:(0+1)*icycle]+xx[0] for i in range(ny)]
    umat=np.vstack(uu)[:]
    vv=[yy[(i)*icycle:(i+1)*icycle][:] for i in range(ny)]
    vmat=np.vstack(vv)[:]
    plt.figure(figsize=(24,12))
    # plt.plot(vmat.T)
    if(dicycle>0):
        ii=dicycle//2;
        plt.plot(umat[:,ii:-ii].T,vmat[:,ii:-ii].T,'.');
    else:
        # ii=1;
        plt.plot(umat[:,:].T,vmat[:,:].T,'.');
    plt.title(fname0)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.savefig('samples_'+fname0[-6:-4]+'.png')
    # plt.show()
    np.savetxt('sample_times_'+fname0,umat)
    np.savetxt('sample_matrix_'+fname0,vmat)

vv=[yy[i*icycle:(i+1)*icycle] for i in range(yy.shape[0]//icycle)]
vmat=np.vstack(vv)[:]
plt.figure()
plt.plot(umat.T,vmat.T)
# plt.plot(umat.T,vmat.T)
plt.show()

