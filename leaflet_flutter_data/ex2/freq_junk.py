# max_freq = (1/fx[iymax])

# print('max_freq',max_freq, fx[iymax])


# inz = np.nonzero(yy)[0]
# y0 = yy[inz] 
# y0 /= y0.max()
# x0 = xx[inz]

# plt.figure()
# plt.plot(x0,y0)
# plt.show()

# plt.figure()afy = np.abs(scipy.fft(y0))

# plt.plot(x0[1:],np.diff(inz))
# plt.show()

# idnz = np.where(np.abs(np.diff(inz))>1)[0]
# plt.figure()
# plt.plot(x0,y0)
# plt.plot(x0[idnz],y0[idnz],'r.')
# plt.show()



# NFFT = 512*4;
# FPS = 5000;
# PixelsPerCm = 100;
# PulseFreq = FPS*( np.arange(0,(NFFT/2)))/NFFT;
# PulseFreq

# ncycles=13
# np.mod(y0.shape[0],ncycles)

# icycle

# icycle = y0.shape[0]//ncycles
# plt.plot(y0[:icycle]);plt.show()
# vv = y0[:icycle*ncycles].reshape(ncycles,icycle)
# uu = x0[:icycle*ncycles].reshape(ncycles,icycle)
# # uu = uu[vv.nonzero()]
# # vv = vv[vv.nonzero()]
# plt.plot(uu.T,vv.T);plt.show()
# plt.plot(vv.T);plt.show()
# vv.nonzero()

# vv.shape




# idnz = np.where(np.abs(np.diff(yy))>0 )
# icycle=np.diff(idnz).max()
# # icycle = int(np.diff(xx[idnz]).min()/dx)
# i1=idnz[0]
# i2=idnz[1]
# imax = i2-i1
# imax = icycle
# jmax = yy.shape[0]//imax
# kmax = jmax * imax
# ncycles = int(np.floor(xx.shape[0] / icycle)-1)
# # x0[:icycle].shape[0]-2

# imax = icycle*ncycles
# print(ncycles,imax)

# uu = xx.copy()
# vv = yy.copy()
# uu=np.mod(uu[:imax],uu[imax])
# uu = uu.reshape(icycle,ncycles)
# vv = vv[:imax]
# vv = vv.reshape(icycle,ncycles)
# plt.plot(vv[:icycle]);plt.show()
# plt.plot(yy[:icycle]);plt.show()

# plt.figure()
# plt.plot(vv[:,0],'.')
# plt.show()

# np.repeat(xx[:imax],[imax,ncycles])
# uu

# u0 = 
# v0 = yy[:kmax]
# v0 = v0.reshape([imax,jmax])
# v0.shape

# plt.figure()
# plt.plot(u0,v0)
# plt.show()


# # zz = np.abs(np.diff(y0))
# zz = np.abs(np.diff(yy))
# i0 = np.where(zz==0)[0]
# print(i0.shape)
# plt.figure()
# plt.plot(xx[i0],yy[i0])
# plt.show()


# plt.figure()
# plt.plot(xx,yy)
# plt.plot(xx[i0],yy[i0],'r.')
# plt.show()


# pmin,pmax = np.percentile(zz,1), np.percentile(zz,99)
# pmin = np.min([np.abs(pmin),np.abs(pmax)])

# np.where()
# plt.figure()
# plt.plot(zz)
# plt.ylim([-pmin,pmin])
# plt.show()


# # plt.figure()
# # dxx = xx[1:]
# # ddxx = xx[2:]
# # dyy = np.diff(yy)
# # ddyy = np.diff(dyy)
# # plt.figure()
# # plt.plot(xx, yy)
# # plt.plot(dxx[:], dyy, 'ro')
# # plt.plot(ddxx[:], ddyy, 'g.')
# # plt.show()


# # np.where()
# zz = np.abs(np.diff(yy))
