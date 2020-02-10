
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
# dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
dname=cwd+'/save/'
# dname=cwd+'/'
fnames = os.listdir(dname)
fnames0 = 'OpenAreaPerimountWaterbpm'
fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]

# bpm,iframe,ifframe=np.loadtxt('bpmdata.txt',unpack=True)
bpmdatas = np.loadtxt(dname+'bpmdata.txt', unpack=False)
bpmdatas = np.array(bpmdatas, dtype=int)
print(bpmdatas)

for fname0 in fnames:
    print(fname0,'\n\n')
    # fname0 = 'OpenAreaPerimountWaterbpm60.txt'
    # for fname0[-6:-4],fname0 in enumerate(fnames[:-2]):
    # for fname0[-6:-4], fname0 in enumerate(fnames):
    istart = 10
    ifinal = 10
    data = np.loadtxt(dname+'datamatrix_'+fname0)[:, istart:-ifinal]


    meanfunc = data.mean(axis=0)

    nt = np.max(data.shape)

    dt = 1/3/10
    t0, tf = 0, nt*dt
    times = np.linspace(t0, tf, nt)


    cdata = data-meanfunc
    varfunc = np.var(cdata, axis=0)
    varfunc = np.sqrt(np.std(cdata, axis=0))
    varfunc.shape
    meanfunc.shape
    ub = varfunc * .005
    lb = varfunc * .005

    plt.figure(figsize=(24,12))
    plt.plot(times,data.T,'k.',alpha=0.2);
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Area vs. Time')
    plt.savefig('data_'+fname0[:-4]+'.png')
    plt.close()
    
    plt.figure(figsize=(24,12))
    plt.plot(times,cdata.T,'k.',alpha=0.2);
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Area vs. Time')
    plt.savefig('centered_'+fname0[:-4]+'.png')
    plt.close()
    

    # plt.figure(figsize=(24,12))
    # plt.fill_between(times,meanfunc,meanfunc+ub)
    # plt.fill_between(times,meanfunc,meanfunc-lb)
    # plt.plot(times,meanfunc)
    # # plt.fill_between(times,meanfunc-varfunc,meanfunc+varfunc)


    plt.figure(figsize=(24,12))
    plt.fill_between(times,meanfunc,meanfunc+ub)
    plt.fill_between(times,meanfunc,meanfunc-lb)
    plt.plot(times,meanfunc)
    # plt.fill_between(times,meanfunc-varfunc,meanfunc+varfunc)
    plt.plot(times,data.T,'k.',alpha=0.2);
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Area vs. Time')
    plt.savefig('variance_'+fname0[:-4]+'.png')
    plt.close()


    plt.figure(figsize=(24, 12))
    # plt.fill_between(times,0*meanfunc,meanfunc+ub,alpha=0.5)
    # plt.fill_between(times,meanfunc,meanfunc-lb,alpha=0.5)
    plt.plot(times, meanfunc)
    plt.fill_between(times, meanfunc-varfunc, meanfunc+varfunc, alpha=0.5)
    plt.plot(times, data.T, 'k.', alpha=0.2)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Area vs. Time')
    plt.savefig('areavstime_'+fname0[:-4]+'.png')
    plt.close()
        
        

    plt.figure(figsize=(24, 12))
    # plt.fill_between(times,0*meanfunc,meanfunc+ub,alpha=0.5)
    # plt.fill_between(times,meanfunc,meanfunc-lb,alpha=0.5)
    plt.plot(times, meanfunc)
    plt.plot(times, cdata.T, 'k.', alpha=0.2)

    for row in cdata.T:
        plt.fill_between(times, row-varfunc, row+varfunc, alpha=0.9)
        # plt.plot(times, .T, 'k.', alpha=0.2)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Area vs. Time')
    plt.savefig('var_areavstime_'+fname0[:-4]+'.png')
    plt.show()
        

    plt.figure(figsize=(24, 12))
    # plt.fill_between(times,0*meanfunc,meanfunc+ub,alpha=0.5)
    # plt.fill_between(times,meanfunc,meanfunc-lb,alpha=0.5)
    plt.fill_between(times, -varfunc, varfunc, alpha=0.5)
    plt.plot(times, cdata.T, 'k.', alpha=0.2)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Area')
    plt.title('Area vs. Time')
    plt.savefig('centeredareavstime_'+fname0[:-4]+'.png')
    plt.close()
    


nr = min(data.shape)
nr2 = 4
pca = PCA(n_components=nr, whiten=True)  # min(df.shape))
X = cdata.copy()

if(dimreductiontype == 'pca'):
    pca = PCA(n_components=nr, whiten=True)  # min(df.shape))
elif(dimreductiontype == 'kpca'):
    pca = KernelPCA(n_components=min(df.shape))
elif(dimreductiontype == 'fa'):
    pca = FactorAnalysis(n_components=min(df.shape))

Z = pca.fit_transform(X)


try:
    print("pca.n_components ", pca.n_components)
    print("pca.n_features_ ", pca.n_features_)
    print("pca.n_samples_ ", pca.n_samples_)
    print('pca.noise_variance_ ', pca.noise_variance_)
except Exception:
    1

try:
    ax, fig = plt.subplots(1, 1)
    plt.plot(pca.explained_variance_ratio_, '-o', ms=4)
    plt.grid()
    plt.title('Variance Explained (Percent) by Component')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.grid()
#    plt.legend(ilabel)
    plt.savefig(cwd+"/"+fname0+'_'+str(fname0[-6:-4])+'_' +
                dimreductiontype+"_"+"explained_variance_ratio_"+".png")
    plt.close()
except Exception:
    1


#pca = FactorAnalysis(n_components=min(df.shape))
#Z = pca.fit_transform(X)
#plt.plot(times,favar)
#plt.title('Variance Explained (Percent) by Component')
#plt.xlabel('Principal Component')
#plt.ylabel('Variance Explained')
#plt.grid()
#plt.savefig(cwd+"/"+fname0+'_'+str(fname0[-6:-4])+'_'+dimreductiontype+"_"+"explained_variance_ratio_"+".png")
##plt.close()

#
#pca = FactorAnalysis(n_components=min(df.shape))
#Z = pca.fit_transform(X)
#favar = pca.noise_variance_
#favar = np.sqrt(favar)
#scale_factor = 8
#plt.figure()
#
#plt.plot(times,X.T ,'b.',ms=1)
#plt.plot(times,Xm[0] ,'k-',lw=6,alpha=0.4)
#plt.plot(times,Xm[0] + scale_factor * favar[:],'g-')
#plt.plot(times,Xm[0] - scale_factor * favar[:],'r-')
#plt.title('Variance Explained (Percent) by Component')
#plt.xlabel('Principal Component')
#plt.ylabel('Variance Explained')
#plt.grid()
#plt.savefig(cwd+"/"+fname0+'_'+str(fname0[-6:-4])+'_'+dimreductiontype+"_"+"bands_"+".png")
##plt.close()
#


try:
    for iy in range(0, nr2):
        #            ax,fig=plt.subplots(1,1)
        x = times
        y = pca.components_[iy]
        plt.figure()
        plt.plot(x, y, '-o', ms=1,alpha=0.6)
    #            for ic, vc in enumerate((iclass)):
    #                plt.plot(x[vc],y[vc],icolor[ic]+'o',label=ilabel[ic])
        plt.grid(which='both')
        plt.xlabel('Time')
        plt.ylabel('Principal Mode '+str(iy))
        plt.savefig(cwd+"/"+fname0+'_'+str(fname0[-6:-4])+'_' +
                    dimreductiontype+"_"+"pm"+str(iy)+".png")

        #plt.close()
except Exception:
    1


try:
    for ix in range(0, nr2):
        for iy in range(0, ix):
            #            ax,fig=plt.subplots(1,1)
            x = pca.components_[ix]
            y = pca.components_[iy]
            plt.figure()
            plt.plot(x, y, '-o', ms=1,alpha=0.6)
        #            for ic, vc in enumerate((iclass)):
        #                plt.plot(x[vc],y[vc],icolor[ic]+'o',label=ilabel[ic])
            plt.grid(which='both')
            plt.xlabel('Principal Mode '+str(ix))
            plt.ylabel('Principal Mode '+str(iy))
            plt.savefig(
                cwd+"/"+str(fname0[-6:-4])+'_'+dimreductiontype+"_"+"pm"+str(ix)+'_'+str(iy)+".png")

            #plt.close()
except Exception:
    1


try:
    plt.figure()
    plt.plot(times, pca.mean_)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Signal Mean')
    plt.savefig(cwd+"/"+fname0+'_'+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.close()
except Exception:
    1


def reconstruction_error(pca, Z, X, pnorm=2, ax=0):
    Xr = pca.inverse_transform(Z)
    resid = Xr-X
    if(pnorm == 'avg'):
        abserr = resid.mean(axis=0)
        relerr = abserr / pca.mean_
    else:
        abserr = np.linalg.norm(resid, ord=pnorm, axis=ax)
        norm = np.linalg.norm(X, ord=pnorm, axis=ax)
        relerr = abserr/norm
    return Xr.T, abserr, relerr


#recon,abserr, relerr = reconstruction_error(pca,Z,X, pnorm='avg')
recon, abserr, relerr = reconstruction_error(pca, Z, X, pnorm=2)
try:
    plt.figure()
    plt.plot(times, mean, 'k-', lw=8, alpha=0.9, label='mean')
    plt.plot(times, recon[:, 0], 'r.', ms=1, alpha=0.8, label='reconstruction')
    plt.plot(times, recon, 'r.', ms=1, alpha=0.2)  # ,label='reconstruction')
    plt.grid()
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Approximate Reconstruction of Signal')
    plt.savefig(cwd+"/"+fname0+'_'+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.close()
except Exception:
    1


#Xr = pca.inverse_transform(pca.transform(mean[None,:]))[0];plt.plot(Xr-mean)
try:
    plt.figure()
    plt.plot(times, relerr, 'r.')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Signal Reconstruction Error')
    plt.title('Relative Signal Reconstruction Error')
    plt.savefig(cwd+"/"+fname0+'_'+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.close()
except Exception:
    1

try:
    plt.figure()
#    plt.plot(times,pca.mean_,label='Mean')
    plt.plot(times, abserr, 'r.', label='Absolute Error')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Signal Reconstruction Error')
    plt.title('Absolute Signal Reconstruction Error')
    plt.savefig(cwd+"/"+fname0+'_'+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.close()
except Exception:
    1


tt = times
Xm = X.mean(axis=0)
Xm = Xm[None, :]
Xc = X-Xm

plt.figure()
plt.plot(times, Xm[0], 'k-', ms=1)
plt.plot(times, X.T, '.', ms=1)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Signal')
plt.title('Signal and Mean')
plt.savefig(cwd+"/"+fname0+'_'+dimreductiontype+'_'+fname0[-6:-4]+'.png')
#plt.close()

plt.figure()
plt.plot(times, Xc.T)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Fluctuation in Signal about Mean')
plt.title('Fluctuation in Signal about Mean')
plt.savefig(cwd+"/"+fname0+'_'+dimreductiontype+'_'+fname0[-6:-4]+'.png')
#plt.close()


plt.figure()
plt.hist(Xc.flatten(), Xc.shape[0], normed=True)
plt.grid()
plt.ylabel('PMF')
plt.xlabel('Fluctuation in Signal about Mean')
plt.title('Fluctuation in Signal about Mean')
plt.savefig(cwd+"/"+fname0+'_'+dimreductiontype+'_'+fname0[-6:-4]+'.png')
#plt.close()
