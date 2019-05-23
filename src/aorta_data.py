#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 10 01:15:02 2019

@author: chaztikov
"""

import os;import numpy as np;import pandas as pd
import os,sys,re,subprocess
import pandas as pd
import numpy as np
import scipy
import scipy.integrate
from scipy.spatial import KDTree
from scipy.interpolate import BSpline
from scipy.interpolate import splrep, splder, sproot,make_interp_spline
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn.decomposition
from sklearn.decomposition import PCA

cwd = os.getcwd()
dname = cwd+'/../data/'
fnames = os.listdir(dname)

for ifname0,fname0 in enumerate(fnames[:-2]):
    fname = dname+fname0
            
    df = pd.read_csv(fname)
    print(df.columns)
    print(df.shape)
    
    
    HR=1
    npeaks = 13
    phi0 = 14150-1
#    phi0 = 0
    phi0 = int(phi0)
    ntau = int(5)
    #ntau = 10
    ntau = int(ntau)
    tau= int(60/HR)
    
    xx = df.values[:,0]
    yy = df.values[:,1]
    
    plt.figure()
    plt.plot(xx,yy,'b')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Raw Signal')
    plt.title(fname0)
    plt.savefig('raw_'+str(ifname0)+'.png')
    plt.show()
      
    xx = df.values[phi0:,0]
    yy = df.values[phi0:,1]
    dyy = np.diff(yy)
    
    nbins = np.sqrt(yy.shape[0] * 1 ).astype(int)  
    inz = np.where(yy>0)[0]
    idnz = np.where(np.abs(dyy)>0)[0]
    dyynz = dyy[idnz]
    dyynz = dyy
    pdc = np.percentile(np.abs(dyynz),99.9)
    iddc = np.where(dyynz>pdc )
    peaks = np.sort(np.abs(dyy))[::-1][:2*npeaks]
    ipeaks = np.argsort(np.abs(dyy))[::-1][:2*npeaks]
    #ipeaks = np.argsort(np.abs(dyy))[::-1][:npeaks]
    iipeaks = np.where(yy[ipeaks]>1e-6)[0]
    inzpeaks = ipeaks[iipeaks]+1
    inzpeaks = np.sort(inzpeaks)
    #these are endpoints of interval
    #pair these with the start points of signal intervals, marked by izpeaks
    iizpeaks = np.where( np.isclose(yy[ipeaks], 0) )[0]
    izpeaks = ipeaks[iizpeaks]
    izpeaks = np.sort(izpeaks)
    
    #cycles and lengths
    icycle = np.array(list(zip(izpeaks,inzpeaks)))
    minclen=np.min(np.diff(icycle,1))
    maxclen=np.max(np.diff(icycle,1))
    padclen = maxclen-np.diff(icycle,1)[:,0]
    padclen = minclen-np.diff(icycle,1)[:,0] 
    icycle[:,1]+=padclen
    times = np.vstack([xx[c[0]:c[1]] for c in icycle]).T
    times -= times[0]
#    times = xx[icycle][:,0][:,None] - xx[icycle]
    output = np.stack([yy[c[0]:c[1]] for c in icycle]).T

    plt.figure()
    plt.plot(xx,yy,'b')
    plt.plot(xx[icycle],yy[icycle],'r.')
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Truncated Raw Signal')
    plt.title(fname0)
    plt.savefig('truncraw_'+str(ifname0)+'.png')
    plt.show()
    
    p1,p2=0,100
    p1,p2=np.percentile(yy[inz],p1),np.percentile(yy[inz],p2)
    plt.figure()
    plt.hist(yy[inz],bins=nbins,normed=True)
    plt.xlim(p1,p2)
    plt.grid()
    plt.ylabel('pmf')
    plt.xlabel('output')
    plt.title('Raw, Nonzero Signal Histogram')
    plt.savefig('histnz_'+str(ifname0)+'.png')
    plt.show()
    
    
    mean = output.mean(axis=1)
    centered = output-mean[:,None]
    plt.figure()
    plt.plot(times,mean,'k-',lw=8,alpha=0.8,label='mean')
    plt.plot(times,output,'b.',ms=2,alpha=0.4)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('output')
    plt.title('Signal Cycles as Samples')
    plt.savefig('mean_'+str(ifname0)+'.png')
    plt.show()
    
    
    plt.figure()
    #plt.plot(times,mean,'k-',lw=4,label='mean')
    plt.plot(times,centered ,'.',ms=1,alpha=0.4)
    plt.grid()
    plt.xlabel('time')
    plt.ylabel('output')
    plt.title('Signal (Centered by Sample Mean)')
    plt.savefig('centered_'+str(ifname0)+'.png')
    plt.show()
    
    
    
        
    
    
    
    X = output.copy().T
    #X = centered.copy().T
    
    nr = X.shape[0]
    dimreductiontype='pca'
    
    from sklearn.decomposition import PCA,KernelPCA,FactorAnalysis
    
    
    if(dimreductiontype=='pca'):
        pca = PCA(n_components = nr ,whiten=True)#min(df.shape))
    elif(dimreductiontype=='kpca'):
        pca = KernelPCA(n_components=min(df.shape))
    elif(dimreductiontype=='fa'):
        pca = FactorAnalysis(n_components=min(df.shape))
        
    Z = pca.fit_transform(X)
    
    try:
        print("pca.n_components ", pca.n_components)
        print("pca.n_features_ ", pca.n_features_)
        print("pca.n_samples_ ", pca.n_samples_)
        print('pca.noise_variance_ ', pca.noise_variance_)
    except Exception:
        1;
    
    try:
        ax,fig=plt.subplots(1,1)
        plt.plot(pca.explained_variance_ratio_,'-o',ms=4)
        plt.grid()
        plt.title('Variance Explained (Percent) by Component')
        plt.xlabel('Principal Component')
        plt.ylabel('Variance Explained')
        plt.grid()
    #    plt.legend(ilabel)
        plt.savefig(cwd+"/"+str(ifname0)+'_'+dimreductiontype+"_"+"explained_variance_ratio_"+".png")
        plt.show()
    except Exception:
        1;
        
        
    #pca = FactorAnalysis(n_components=min(df.shape))
    #Z = pca.fit_transform(X)
    #plt.plot(times[:,0],favar)
    #plt.title('Variance Explained (Percent) by Component')
    #plt.xlabel('Principal Component')
    #plt.ylabel('Variance Explained')
    #plt.grid()
    #plt.savefig(cwd+"/"+str(ifname0)+'_'+dimreductiontype+"_"+"explained_variance_ratio_"+".png")
    #plt.show()
    
    #
    #pca = FactorAnalysis(n_components=min(df.shape))
    #Z = pca.fit_transform(X)
    #favar = pca.noise_variance_
    #favar = np.sqrt(favar)
    #scale_factor = 8
    #plt.figure()
    #
    #plt.plot(times[:,0],X.T ,'b.',ms=1)
    #plt.plot(times[:,0],Xm[0] ,'k-',lw=6,alpha=0.4)
    #plt.plot(times[:,0],Xm[0] + scale_factor * favar[:],'g-')
    #plt.plot(times[:,0],Xm[0] - scale_factor * favar[:],'r-')
    #plt.title('Variance Explained (Percent) by Component')
    #plt.xlabel('Principal Component')
    #plt.ylabel('Variance Explained')
    #plt.grid()
    #plt.savefig(cwd+"/"+str(ifname0)+'_'+dimreductiontype+"_"+"bands_"+".png")
    #plt.show()
    #
    
    
    try:
        for iy in range(0,nr):
    #            ax,fig=plt.subplots(1,1)
            x = times
            y = pca.components_[iy]
            plt.figure()
            plt.plot(x,y,'o',ms=4)
    #            for ic, vc in enumerate((iclass)):
    #                plt.plot(x[vc],y[vc],icolor[ic]+'o',label=ilabel[ic])
            plt.grid(which='both')
            plt.xlabel('Time')
            plt.ylabel('Principal Mode '+str(iy))
            plt.savefig(cwd+"/"+str(ifname0)+'_'+dimreductiontype+"_"+"pm"+str(ix)+"pm"+str(iy)+".png")
    
            plt.show()
    except Exception:
        1;
    
    
    
    try:
        plt.figure()
        plt.plot(times,pca.mean_)
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Signal Mean')
        plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0+'.png')
        plt.show()
    except Exception:
        1;
    
    def reconstruction_error(pca,Z,X,pnorm=2,ax=0):
        Xr = pca.inverse_transform(Z)
        resid = Xr-X
        if(pnorm=='avg'):
            abserr = resid.mean(axis=0)
            relerr = abserr / pca.mean_
        else:
            abserr = np.linalg.norm(resid,ord=pnorm,axis=ax)
            norm = np.linalg.norm(X,ord=pnorm,axis=ax)
            relerr = abserr/norm
        return Xr.T, abserr, relerr
    #recon,abserr, relerr = reconstruction_error(pca,Z,X, pnorm='avg')
    recon,abserr, relerr = reconstruction_error(pca,Z,X, pnorm=2)
    try:
        plt.figure()
        plt.plot(times[:,0],mean,'k-',lw=8,alpha=0.9,label='mean')
        plt.plot(times[:,0],recon[:,0],'r.',ms=1,alpha=0.8,label='reconstruction')
        plt.plot(times,recon,'r.',ms=1,alpha=0.2)#,label='reconstruction')
        plt.grid()
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Approximate Reconstruction of Signal')
        plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0+'.png')
        plt.show()
    except Exception:
        1;
    
    
    #Xr = pca.inverse_transform(pca.transform(mean[None,:]))[0];plt.plot(Xr-mean)
    try:
        plt.figure()
        plt.plot(times,relerr,'r.')
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Signal Reconstruction Error')
        plt.title('Relative Signal Reconstruction Error')
        plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0+'.png')
        plt.show()
    except Exception:
        1;
        
    try:
        plt.figure()
    #    plt.plot(times,pca.mean_,label='Mean')
        plt.plot(times,abserr,'r.',label='Absolute Error')
        plt.grid()
        plt.xlabel('Time')
        plt.ylabel('Signal Reconstruction Error')
        plt.title('Absolute Signal Reconstruction Error')
        plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0+'.png')
        plt.show()
    except Exception:
        1;
    
    
    tt = times[:,0]
    Xm = X.mean(axis=0)
    Xm = Xm[None,:]
    Xc = X-Xm
    
    plt.figure()
    plt.plot(times,Xm[0],'k-',ms=1)
    plt.plot(times,X.T,'.',ms=1)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Signal')
    plt.title('Signal and Mean')
    plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0+'.png')
    plt.show()
    
    plt.figure()
    plt.plot(times[:,0], Xc.T)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Fluctuation in Signal about Mean')
    plt.title('Fluctuation in Signal about Mean')
    plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0+'.png')
    plt.show()
    
    
    plt.figure()
    plt.hist(Xc.flatten(),Xc.shape[0],normed=True)
    plt.grid()
    plt.ylabel('PMF')
    plt.xlabel('Fluctuation in Signal about Mean')
    plt.title('Fluctuation in Signal about Mean')
    plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0+'.png')
    plt.show()
    
    
    #    U,S,V = np.linalg.svd(Xc,full_matrices=False)
    #    
    #    plt.plot(S,'-o')
    #    explained_variance = np.cumsum(S)/np.sum(S,axis=0)
    #    plt.plot(explained_variance,'-o')
    #    plt.show()
    
    
    #tol = 0.3
    #itrunc = np.where(explained_variance>tol)[0].min()
    #
    #for itrunc in range(S.shape[0], S.shape[0]-1,-1):
    #    S[itrunc:]*=0
    #    Xrc = U.dot(np.diag(S).dot(V))
    #    
    #    error = Xc-Xrc
    #    terror = np.mean(Xrc-Xc,axis=0)
    #    serror = np.linalg.norm(Xrc-Xc,axis=1,ord=2)
    #    nbins = np.sqrt(2 * serror.shape[0]).astype(int)
    ##    print( terror)
    ##    print('itrunc', itrunc, '' ,' Signal Variance ', Xc.var() - Xrc.var() , ' Signal Fraction ', 1 - Xrc.var() / Xc.var() )
    #    xstdev = np.sqrt(np.var(error,axis=1))
    #    xtimevariation = np.sqrt(np.var(error,axis=0))
    #    print('itrunc', itrunc, '' ,' Signal Time Variation ', xtimevariation,'Sample StDev ', xstdev )#, ' SNR ', np.sqrt( Xc.var() / Xrc.var() - 1 ) )
    #    
    #    plt.figure()
    #    plt.plot(tt,terror,'.')
    #    plt.show()
    #    
    #    plt.figure()
    #    plt.plot(times, error.T,'.',ms=2,alpha=0.2)
    #    plt.plot(tt, error[0],'.',ms=2,alpha=0.2)
    #    plt.show()
    #    
    #    plt.figure()
    #    plt.hist(serror,bins=nbins)
    #    plt.show()
