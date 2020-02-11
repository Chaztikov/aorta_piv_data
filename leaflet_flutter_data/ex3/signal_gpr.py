
# import os
# import sys
# import re
# import numpy as np
# import pandas as pd
# import subprocess
# import pandas as pd

# import numpy as np
# import numpy.fft as fft
# from numpy.fft import fftfreq


# import scipy
# import scipy.integrate


# from scipy.spatial import KDTree
# from scipy.interpolate import BSpline
# from scipy.interpolate import splrep, splder, sproot, make_interp_spline
# import scipy.sparse.linalg as spla
# import matplotlib.pyplot as plt
# import os
# import sys
# import re
# import numpy as np
# import pandas as pd
# import subprocess
# import pandas as pd
# import numpy as np
# import scipy
# import scipy.integrate
# from scipy.spatial import KDTree
# from scipy.interpolate import BSpline
# from scipy.interpolate import splrep, splder, sproot, make_interp_spline
# import scipy.sparse.linalg as spla
# import matplotlib.pyplot as plt

# # import seaborn as sns
# import sklearn.decomposition
# from sklearn.decomposition import PCA
# from sklearn.decomposition import PCA, KernelPCA, FactorAnalysis
# dimreductiontype = 'pca'

# interpolate_signal = 1

# cwd = os.getcwd()
# # dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
# dname=cwd+'/save/'
# # dname=cwd+'/'
# fnames = os.listdir(dname)
# fnames0 = 'OpenAreaPerimountWaterbpm'
# fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]


# # bpm,iframe,ifframe=np.loadtxt('bpmdata.txt',unpack=True)
# bpmdatas = np.array(
#     np.loadtxt(dname+'bpmdata.txt', unpack=False)
#     , dtype=int)

# for fname0 in fnames[:1]:
#     print(fname0,'\n\n')
#     # fname0 = 'OpenAreaPerimountWaterbpm60.txt'
#     # for fname0[-6:-4],fname0 in enumerate(fnames[:-2]):
#     # for fname0[-6:-4], fname0 in enumerate(fnames):
#     istart = 10
#     ifinal = 10
#     try:
#         data = np.loadtxt(dname+'datamatrix_'+fname0)[:, istart:-ifinal]
#     except Exception:
#         # df=pd.DataFrame()
#         df=pd.read_csv(dname+'list_matrix_'+fname0)
#         data=df.values
#         data[np.isnan(data)]=0;



from mpl_toolkits.mplot3d import Axes3D
import time,sys,os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
# 0 = all messages are logged (default behavior)
# 1 = INFO messages are not printed
# 2 = INFO and WARNING messages are not printed
# 3 = INFO, WARNING, and ERROR messages are not printed


# import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp

import sklearn
import sklearn.model_selection


tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

os.system('rm *.txt')
os.system('rm ./save/*.txt')
os.system('rm ./png/*.png')



# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'


data_dir = '/home/chaztikov/Documents/work/uq/data/'
fname = 'aortic_valve_piv_data.txt'
fname = 'pericardial_valve_piv_data.txt'

# atrial_pressure upstream_pressure pump_flow flow downstream_pressure
label1, label2 = 'atrial_pressure', 'upstream_pressure'
label1 = 'flow'

aortic_labels = ['upstream_pressure' 'flow' 'downstream_pressure' 'pdva']
pericardial_labels = [
    'atrial_pressure' 'upstream_pressure' 'pump_flow' 'flow' 'downstream_pressure']

kernel_type = 'ExponentiatedQuadratic'
ii0, dii, iif = 0, 10, 2650

dii = 2

num_optimizer_iters = 10000

num_predictive_samples = 10

NUM_TRAINING_POINTS = 0

BPM = 60
InitialFrame = 12000
StartFFTFrame = 2000
NFFT = 512*4
FPS = 5000
PixelsPerCm = 100


FramesPerPulse = 60/BPM*FPS
# Area = A{1,2}/(PixelsPerCm^2);
# Time =(1:FPS)/FPS*1000;
# Pulses = floor((size(Area,1)-InitialFrame)/FramesPerPulse);

filenames = ['aortic_valve_piv_data.csv','pericar_valve_piv_data.csv']
times = np.arange(0, 0.855, 0.000334)

paramslist = []
header = ['amplitude','length_scale','observation_noise_variance']


dname = '/home/chaztikov/git/aorta_piv_data/data/original/'
fnames = os.listdir(dname)
fnames0 = 'OpenAreaPerimountWaterbpm'
fnames = [fnames0+str(i)+'.txt' for i in [60, 80, 100, 120]]

bpmdatas = np.array(
    np.loadtxt(dname+'bpmdata.txt', unpack=False)
    , dtype=int)

for fname0 in fnames[:1]:
    print(fname0,'\n\n')
    data = np.loadtxt(dname+fname0)[ :, :]

    train, test = sklearn.model_selection.train_test_split( data, test_size=int(data.shape[0]*.70) )#, np.arange(iif,2), np.arange(iif,2))
    valid, test = sklearn.model_selection.train_test_split( test, test_size=int(test.shape[0]*.50) )#, np.arange(iif,2), np.arange(iif,2))

    
    x,y = train[:,0],train[:,1]
        
    inz = np.nonzero(y)[0]
    x=x[inz]
    y=y[inz]

    x = np.atleast_2d(x)

    observation_index_points_ = x #times
    observations_ = y #df[val].values

    # observation_index_points_ = np.array(observation_index_points_, np.float64)
    # observation_index_points_ = observation_index_points_[::dii]
    # observations_ = observations_[::dii]
    
    checkpoints_iterator_ = tf.train.checkpoints_iterator('.')

    def build_gp(amplitude, length_scale, observation_noise_variance):
        # mean_fn = None
        kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

        model = tfd.GaussianProcess(
            kernel=kernel,
            index_points=observation_index_points_,
            observation_noise_variance=observation_noise_variance)
            # mean_fn=mean_fn,

        return model

    gp_joint_model = tfd.JointDistributionNamed({
        'amplitude': tfd.Normal(loc=400., scale=np.float64(1.)),
        'length_scale': tfd.Normal(loc=0.03, scale=np.float64(1.)),
        'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
        'observations': build_gp,
    })

    sample = gp_joint_model.sample()
    lp = gp_joint_model.log_prob(sample)

    print("sampled {}".format(sample))
    print("log_prob of sample: {}".format(lp))
    
    

    constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

    length_scale0 = np.abs(np.diff(x)).min() #0.01,
    amplitude0 = y.max()-y.min()
    observation_noise_variance0 = np.std(y)/np.mean(y)
    # np.std(y)
    # params

    amplitude_var = tfp.util.TransformedVariable(
        initial_value = amplitude0,
        bijector=constrain_positive,
        name='amplitude',
        dtype=np.float64)

    length_scale_var = tfp.util.TransformedVariable(
        initial_value = length_scale0, 
        bijector=constrain_positive,
        name='length_scale',
        dtype=np.float64)

    observation_noise_variance_var = tfp.util.TransformedVariable(
        initial_value = observation_noise_variance0,
        bijector=constrain_positive,
        name='observation_noise_variance',
        dtype=np.float64)

    trainable_variables = [v.trainable_variables[0] for v in
                        [amplitude_var,
                            length_scale_var,
                            observation_noise_variance_var]]

    # Use `tf.function` to trace the loss for more efficient evaluation.
    @tf.function(autograph=False, experimental_compile=False)
    def target_log_prob(amplitude, length_scale, observation_noise_variance):
        return gp_joint_model.log_prob({
            'amplitude': amplitude,
            'length_scale': length_scale,
            'observation_noise_variance': observation_noise_variance,
            'observations': observations_
        })
    
    # Now we optimize the model parameters.
    optimizer = tf.optimizers.Adam(learning_rate=.01)

    # Store the likelihood values during training, so we can plot the progress
    # lls_ = np.zeros(num_optimizer_iters, np.float64)
    lls_=[]
    for i in range(num_optimizer_iters):
        with tf.GradientTape() as tape:
            loss = -target_log_prob(amplitude_var,
                                    length_scale_var,
                                    observation_noise_variance_var)
        grads = tape.gradient(loss, trainable_variables)
        optimizer.apply_gradients(zip(grads, trainable_variables))
        lls_.append(loss)
        # lls_[i] = loss

    # # checkpoints_iterator_;
    # checkpoint_ = tf.train.Checkpoint(optimizer=optimizer)
    # manager = tf.train.CheckpointManager(
    #     checkpoint_, './.tf_ckpts', checkpoint_name='checkpoint_'+str(i), max_to_keep=3)

    def write_trained_parameters():
        return [amplitude_var._value().numpy(), length_scale_var._value(
        ).numpy(), observation_noise_variance_var._value().numpy()]
        # return params

    # x = gp_joint_model.sample()
    # lp = gp_joint_model.log_prob(x)

    params = write_trained_parameters()
    paramslist.append(params)
    print(params, '\n')



    loglikelihood = np.array(lls_)
    np.savetxt(filename[:7]+'_loglikelihood.txt', loglikelihood)
    
    np.savetxt('paramslists.txt', paramslist)




        
np.savetxt('paramslists.txt', paramslist)
