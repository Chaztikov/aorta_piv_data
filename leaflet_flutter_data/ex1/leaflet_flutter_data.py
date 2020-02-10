from mpl_toolkits.mplot3d import Axes3D
import time

import numpy as np
import matplotlib.pyplot as plt
# import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_probability as tfp
tfb = tfp.bijectors
tfd = tfp.distributions
tfk = tfp.math.psd_kernels
tf.enable_v2_behavior()

# Configure plot defaults
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['grid.color'] = '#666666'



data_dir = '/home/chaztikov/git/aorta_piv_data/data/original/'
fname = 'OpenAreaPerimountWaterbpm60.txt'


data = np.loadtxt(data_dir+fname)

# A = textscan(fileID,'%f %f','Delimiter',',');

# %Edit BPM
BPM = 80;
InitialFrame =  000;
StartFFTFrame = 1;
NFFT = 512*4;
FPS = 5000;
PixelsPerCm = 100;



kernel_type = 'ExpSinSquared'
ii0, dii, iif = 0, 10, 2650
num_predictive_samples = 10






def build_gp(amplitude, length_scale, period, observation_noise_variance):
  """Defines the conditional dist. of GP outputs, given kernel parameters."""
  # kernel = tfk.ExponentiatedQuadratic(amplitude, length_scale)

  if(kernel_type == 'ExpSinSquared'):
      kernel = tfk.ExpSinSquared(amplitude, length_scale, period)

  # Create the GP prior distribution, which we will use to train the model parameters.
  model = tfd.GaussianProcess(
      kernel=kernel,
      index_points=observation_index_points_,
      observation_noise_variance=observation_noise_variance)

  return model


observation_index_points_, observations_, times = generate_data(
    NUM_TRAINING_POINTS, label1, label2, data_dir, fname)

gp_joint_model = tfd.JointDistributionNamed({
    'amplitude': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'length_scale': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'period': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observation_noise_variance': tfd.LogNormal(loc=0., scale=np.float64(1.)),
    'observations': build_gp,
})


constrain_positive = tfb.Shift(np.finfo(np.float64).tiny)(tfb.Exp())

amplitude_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='amplitude',
    dtype=np.float64)

length_scale_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='length_scale',
    dtype=np.float64)


period_var = tfp.util.TransformedVariable(
    initial_value=0.03,
    bijector=constrain_positive,
    name='period',
    dtype=np.float64)

observation_noise_variance_var = tfp.util.TransformedVariable(
    initial_value=1.,
    bijector=constrain_positive,
    name='observation_noise_variance_var',
    dtype=np.float64)

trainable_variables = [v.trainable_variables[0] for v in
                       [amplitude_var,
                        length_scale_var,
                        period_var,
                        observation_noise_variance_var]]

# Use `tf.function` to trace the loss for more efficient evaluation.
@tf.function(autograph=False, experimental_compile=False)
def target_log_prob(amplitude, length_scale, period, observation_noise_variance):
  return gp_joint_model.log_prob({
      'amplitude': amplitude,
      'length_scale': length_scale,
      'period': period,
      'observation_noise_variance': observation_noise_variance,
      'observations': observations_
  })


# Now we optimize the model parameters.
num_iters = 1000
optimizer = tf.optimizers.Adam(learning_rate=.01)

# Store the likelihood values during training, so we can plot the progress
lls_ = np.zeros(num_iters, np.float64)
for i in range(num_iters):
  with tf.GradientTape() as tape:
    loss = -target_log_prob(amplitude_var,
                            length_scale_var,
                            period_var,
                            observation_noise_variance_var)
  grads = tape.gradient(loss, trainable_variables)
  optimizer.apply_gradients(zip(grads, trainable_variables))
  lls_[i] = loss


# Plot the loss evolution
plt.figure(figsize=(12, 4))
plt.plot(lls_)
plt.grid()
plt.title('Log marginal likelihood')
plt.xlabel("Training iteration")
plt.ylabel("Log marginal likelihood")
plt.savefig('lml_out.png')


'''
USE MODEL
'''

# tf.saved_model.save(
# tf.Variable({
#       'amplitude': amplitude,
#       'length_scale': length_scale,
#       'period': period,
#       'observation_noise_variance': observation_noise_variance,
#       'observations': observations_
#   }))

# Having trained the model, we'd like to sample from the posterior conditioned
# on observations. We'd like the samples to be at points other than the training
# inputs.

# predictive_index_points_ = np.linspace(-1.2, 1.2, 200, dtype=np.float64)

predictive_index_points_ = np.array(times[:], dtype=np.float64)
# Reshape to [200, 1] -- 1 is the dimensionality of the feature space.
predictive_index_points_ = predictive_index_points_[..., np.newaxis]

optimized_kernel = tfk.ExpSinSquared(
    amplitude_var, length_scale_var, period_var)
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
plt.figure(figsize=(12, 4))
plt.scatter(observation_index_points_[:, 0],
            observations_,
            c='b',
            marker='o',
            label='Observations')
for i in range(num_predictive_samples):
  plt.plot(predictive_index_points_,
           samples[i, :],
           c='r', alpha=.1, marker='.',
           label='Posterior Sample' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.savefig('gpm_out.png')


# Plot the true function, observations, and posterior samples.
tau = period_var._value().numpy()
plt.figure(figsize=(12, 4))
plt.scatter(np.mod(observation_index_points_[:, 0], tau),
            observations_,
            c='b',
            marker='o',
            label='Observations')
for i in range(num_predictive_samples):
  plt.scatter(np.mod(predictive_index_points_, tau),
              samples[i, :],
              c='r', alpha=.1, marker='.',
              label='Posterior Sample' if i == 0 else None)
leg = plt.legend(loc='upper right')
for lh in leg.legendHandles:
    lh.set_alpha(1)
plt.xlabel(r"Index points ($\mathbb{R}^1$)")
plt.ylabel("Observation space")
plt.savefig('gpm_periodic_out.png')


def print_trained_parameters():
    print('Trained parameters:')
    print('amplitude: {}'.format(amplitude_var._value().numpy()))
    print('length_scale: {}'.format(length_scale_var._value().numpy()))
    print('period_var: {}'.format(period_var._value().numpy()))
    print('observation_noise_variance: {}'.format(
        observation_noise_variance_var._value().numpy()))

    print('period_var: {}'.format(period_var._value().numpy()))
    print('observation_noise_variance: {}'.format(
        observation_noise_variance_var._value().numpy()))


# checkpoint = tf.train.Checkpoint(optimizer=optimizer)
# manager = tf.train.CheckpointManager(
#     checkpoint, './.tf_ckpts',
#     checkpoint_name=checkpoint_name, max_to_keep=3)

print_trained_parameters()
