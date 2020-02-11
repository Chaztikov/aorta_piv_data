

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

num_predictive_samples = 1

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

ntrain=100
nvalid=100
for fname0 in fnames[:1]:
    print(fname0,'\n\n')


    data = np.loadtxt(dname+fname0)[ : , : ]

    train, test = sklearn.model_selection.train_test_split( data, test_size=int(data.shape[0]*.90) )#, np.arange(iif,2), np.arange(iif,2))
    valid, test = sklearn.model_selection.train_test_split( test, test_size=int(test.shape[0]*.50) )#, np.arange(iif,2), np.arange(iif,2))

    x,y = data[:ntrain,0], data[:ntrain,1]
    
    # x,y = train[:,0], train[:,1]
    # x,y = valid[:,0], valid[:,1]
    # x,y = test[:,0], test[:,1]
        
    inz = np.nonzero(y)[0]
    x=x[inz]
    y=y[inz]

    x = np.atleast_2d(x).T

    observation_index_points_ = x #times
    observations_ = y #df[val].values

    checkpoints_iterator_ = tf.train.checkpoints_iterator('.')

    # def build_gp(amplitude, length_scale, observation_noise_variance):
    #     # mean_fn = None
    #     kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

    #     model = tfd.GaussianProcess(
    #         kernel=kernel,
    #         index_points=observation_index_points_,
    #         observation_noise_variance=observation_noise_variance)
    #         # mean_fn=mean_fn,

    #     return model

    
    params = np.array([414.4366916986194, 0.03000000000000001, 0.36970283292423645] ,dtype=np.float64)

    '''
    USE MODEL
    '''

    x.shape
    x.min()
    x.max()
    nvalid=max(x.shape)
    num_predictive_samples=1
    # predictive_index_points_ = np.linspace(valid[0,0], valid[-1,0], nvalid ) 
    predictive_index_points_ = np.linspace(x.min(), x.max(), nvalid)
    # Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
    predictive_index_points_ = predictive_index_points_[..., np.newaxis]

    amplitude_var, length_scale_var = params[[0,1]]
    observation_noise_variance_var = params[2]
    optimized_kernel = tfk.ExponentiatedQuadratic(amplitude_var, length_scale_var)
    
    gprm = tfd.GaussianProcessRegressionModel(
        kernel=optimized_kernel,
        index_points=predictive_index_points_,
        observation_index_points=observation_index_points_,
        observations=observations_,
        observation_noise_variance=observation_noise_variance_var,
        predictive_noise_variance=0.)

    # Create op to draw  50 independent samples, each of which is a *joint* draw
    # from the posterior at the predictive_index_points_. Since we have 200 input
    # locations as defined above, this posterior distribution over corresponding
    # function values is a 200-dimensional multivariate Gaussian distribution!
    samples = gprm.sample(num_predictive_samples)

    # Plot the true function, observations, and posterior samples.
    plt.figure(figsize=(24, 12))
    plt.scatter(observation_index_points_[:, 0],
                observations_,
                c='b',
                marker='.',
                label='Observations')
    for i in range(num_predictive_samples):
        plt.plot(predictive_index_points_[:,0],
                samples[i,:],
                c='r', alpha=.9, ms=10,
                label='Posterior Sample' if i == 0 else None)
    leg = plt.legend(loc='upper right')
    for lh in leg.legendHandles:
        lh.set_alpha(1)
    plt.grid()
    plt.xlabel(r"Index points ($\mathbb{R}^1$)")
    plt.ylabel("Observation space")
    plt.savefig(fname0+'_gpm_out.png')
    plt.show()
