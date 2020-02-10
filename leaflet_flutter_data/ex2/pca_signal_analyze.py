
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
    plt.savefig(cwd+"/"+str(fname0[-6:-4])+'_' +
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
#plt.savefig(cwd+"/"+str(fname0[-6:-4])+'_'+dimreductiontype+"_"+"explained_variance_ratio_"+".png")
##plt.show()

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
#plt.savefig(cwd+"/"+str(fname0[-6:-4])+'_'+dimreductiontype+"_"+"bands_"+".png")
##plt.show()
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
        plt.savefig(cwd+"/"+str(fname0[-6:-4])+'_' +
                    dimreductiontype+"_"+"pm"+str(iy)+".png")

        #plt.show()
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

            #plt.show()
except Exception:
    1


try:
    plt.figure()
    plt.plot(times, pca.mean_)
    plt.grid()
    plt.xlabel('Time')
    plt.ylabel('Signal Mean')
    plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.show()
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
    plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.show()
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
    plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.show()
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
    plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0[-6:-4]+'.png')
    #plt.show()
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
plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0[-6:-4]+'.png')
#plt.show()

plt.figure()
plt.plot(times, Xc.T)
plt.grid()
plt.xlabel('Time')
plt.ylabel('Fluctuation in Signal about Mean')
plt.title('Fluctuation in Signal about Mean')
plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0[-6:-4]+'.png')
#plt.show()


plt.figure()
plt.hist(Xc.flatten(), Xc.shape[0], normed=True)
plt.grid()
plt.ylabel('PMF')
plt.xlabel('Fluctuation in Signal about Mean')
plt.title('Fluctuation in Signal about Mean')
plt.savefig(cwd+"/"+dimreductiontype+'_'+fname0[-6:-4]+'.png')
#plt.show()
