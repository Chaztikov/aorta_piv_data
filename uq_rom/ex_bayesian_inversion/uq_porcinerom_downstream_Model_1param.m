%% INVERSION: PREDATOR-PREY MODEL CALIBRATION
%
% In this example, the classical predator-prey equations (or Lotka-Volterra 
% equations) are calibrated against a time series that represents the
% relative population sizes of lynxes and Pouts in a region.





%% 1 - INITIALIZE UQLAB
%
% Clear all variables from the workspace, set the random number generator
% for reproducible results, and initialize the UQLab framework:
clear all
clearvars
rng(100,'twister')
uqlab

%%
% Load the measured population size stored in |Data|:
% Data=load('uq_porcineromModel.mat')
Data=load('uq_porcineromModel.mat')

close all

figure(1)
plot(Data.tstep,Data.Pout)
hold on;
plot(Data.tstep,Data.Pin)
hold off;
xlabel('time')
ylabel('pressure')
title('Phase Plot: Pressures ')
grid()
savefig('test1')
close(gcf)

figure(3)
plot(Data.Pin,Data.Pout, '-o')
xlabel('Pressure In')
ylabel('Pressure Out')
title('Phase Plot: Pressures')
grid()
savefig('test2')
close(gcf)

% figure(2)
% plot(Data.tstep,Data.Pout-mean(Data.Pout))
% hold on;
% plot(Data.tstep,Data.Pin-mean(Data.Pin))
% hold off;
% xlabel('time')
% ylabel('pressure')
% title('Phase Plot: Pressures (Centered)')
% grid()


% figure(3)
% plot(Data.Pin-mean(Data.Pin),Data.Pout-mean(Data.Pout), '-o')
% xlabel('Pressure In')
% ylabel('Pressure Out')
% title('Phase Plot: Pressures (Centered)')
% grid()


figure(1)
plot(Data.tstep,Data.pdva)
hold on;
plot(Data.tstep,Data.Pin)
hold off;
xlabel('time')
ylabel('pressure')
title('Phase Plot: Pressures ')
grid()

%% 2 - FORWARD MODEL
%
% The forward model used for calibration is the solution of the 
% Lotka-Volterra differential equations given by:
%
% $$ \frac{\mathrm{d}\,p_{\mathrm{prey}}}{\mathrm{d}\,t}=\R1 p_{\mathrm{prey}} - \R2 p_{\mathrm{prey}}p_{\mathrm{pred}}$$
%
% $$ \frac{\mathrm{d}\,p_{\mathrm{pred}}}{\mathrm{d}\,t}=-\C p_{\mathrm{pred}} + \delta p_{\mathrm{prey}}p_{\mathrm{pred}}$$
%
% These equations describe the evolution over time $t$ of two populations: 
% the _prey_ $p_{\mathrm{prey}}$ and the _predator_ $p_{\mathrm{pred}}$.
% 
% The forward model computes the population sizes for the duration of
% 21 years, for which measurements $y_{\mathrm{prey}}(t)$ and 
% $y_{\mathrm{pred}}(t)$ are available.
% The model takes as input parameters:
%
% # $\R1$: growth rate of the prey population
% # $\R2$: shrinkage rate of the prey population (relative to the product
% of the population sizes)
% # $\C$: shrinkage rate of the predator population
% # $\delta$: growth rate of the predator population (relative to the product
% of the population sizes)
% # $p_{\mathrm{prey},0}$: initial prey population
% # $p_{\mathrm{pred},0}$: initial predator population
%
% The computation is carried out by the function |uq_predatorPreyModel|
% supplied with UQLab. For every set of input parameters, the function
% returns the population evolution in a 21-year time series.

%%
% Shift the year in the loaded data for consistency with the forward model
% (start from 0):
% 
normYear = Data.tstep-Data.tstep(1);

%%

% Specify the forward models as a UQLab MODEL object:

% ModelOpts.mHandle = @(x) uq_porcineromModel(x,normYear);

ModelOpts.mHandle = @(x) uq_porcinerom_downstream_Model(x,normYear);

ModelOpts.isVectorized = true;

myForwardModel = uq_createModel(ModelOpts);

%% 3 - PRIOR DISTRIBUTION OF THE MODEL PARAMETERS
%
% To encode the available information about the model parameters
% $x_{\mathcal{M}}$ before any experimental observations,
% lognormal prior distributions are put on the parameters as follows:
%
% # $\R1 \siData.Pin
% # $\mathcal{LN}(\mu_\R1 = 1, \sigma_\R1 = 0.1)$
% # $\R2 \sim \mathcal{LN}(\mu_\R2 = 5\times10^{-2}, \sigma_\R2 = 5\times10^{-3})$
% # $\C \sim \mathcal{LN}(\mu_\C = 1, \sigma_\C = 0.1)$
% # $\delta \sim \mathcal{LN}(\mu_\delta = 5\times10^{-2}, \sigma_\delta = 5\times10^{-3})$
% # $p_{\mathrm{prey},0} \sim \mathcal{LN}(\lambda_{p_{prey}} = \log{(10)}, \zeta_{p_{prey}} = 1)$
% # $p_{\mathrm{pred},0} \sim \mathcal{LN}(\lambda_{p_{pred}} = \log{(10)}, \zeta_{p_{pred}} = 1)$
%
% Specify these prior distributions as a UQLab INPUT object:
% 
% R1 = X(:,1);
% R2 =  X(:,2);
% C = X(:,3);
% %P0 = X(:,4);
% 
% % Initial conditions
% initialP1 = X(:,5);
% 
% % Measured Function Values
% Pin = X(:,6);
% Pout = X(:,7);
% Q =  X(:,8);
% 
% Cvia=0.1
% R1=0.15
% R2=0.15

nParams=0;
nParams=nParams+1;
PriorOpts.Marginals(nParams).Name = ('R1');
PriorOpts.Marginals(nParams).Type = 'LogNormal';
PriorOpts.Marginals(nParams).Moments = [0.15 .015];

nParams=nParams+1;
PriorOpts.Marginals(nParams).Name = ('R2');
PriorOpts.Marginals(nParams).Type = 'LogNormal';
PriorOpts.Marginals(nParams).Moments = [0.15 0.015];


nParams=nParams+1;
PriorOpts.Marginals(nParams).Name = ('C');
PriorOpts.Marginals(nParams).Type = 'LogNormal';
PriorOpts.Marginals(nParams).Moments = [0.1 0.01];


% nParams=nParams+1;
% PriorOpts.Marginals(nParams).Name = ('P0');
% PriorOpts.Marginals(nParams).Type = 'LogNormal';
% PriorOpts.Marginals(nParams).Moments = [1 0.1];


% nParams=nParams+1;
% PriorOpts.Marginals(nParams).Name = ('initP1');
% PriorOpts.Marginals(nParams).Type = 'LogNormal';
% PriorOpts.Marginals(nParams).Parameters = [log(10) 1]; 


nParams=nParams+1;
PriorOpts.Marginals(nParams).Name = ('Pout');
PriorOpts.Marginals(nParams).Type = 'Normal';
PriorOpts.Marginals(nParams).Parameters = [mean(Data.Pout) std(Data.Pout)];

Data.Pout

% nParams=nParams+1;
% PriorOpts.Marginals(7).Name = ('Pout');
% PriorOpts.Marginals(7).Type = 'LogNormal';
% PriorOpts.Marginals(7).Parameters = [log(10) 1];
% 

% Pdva is the projected dynamic valve area measured for porcine valve. Basically the valve opening area measured in the experiment.


% nParams=nParams+1;
% PriorOpts.Marginals(7).Name = ('PDVA');
% PriorOpts.Marginals(7).Type = 'LogNormal';
% PriorOpts.Marginals(7).Parameters = [log(10) 1];
% 

% nParams=nParams+1;
% PriorOpts.Marginals(8).Name = ('Q');
% PriorOpts.Marginals(8).Type = 'LogNormal';
% PriorOpts.Marginals(8).Parameters = [log(10) 1];

nParams = length(PriorOpts.Marginals)
myPriorDist = uq_createInput(PriorOpts);

%% 4 - MEASUREMENT DATA

myData(1).y = Data.Pout';
myData(1).Name = 'Pout data';
myData(1).MOMap = 1:length(Data.Pout); % Output ID

% myData(2).y = Data.lynx'/1000; %in 1000
% myData(2).Name = 'Lynx data';
% myData(2).MOMap = 22:42; % Output ID

%% 5 - DISCREPANCY MODEL
%
% To infer the discrepancy variance, lognormal priors are put on the
% discrepancy parameters:
%
% * $\sigma^2_{\mathrm{prey}} \sim \mathcal{LN}(\lambda_{\sigma^2_\mathrm{prey}} = -1, \zeta_{\sigma^2_\mathrm{prey}} = 1)$
% * $\sigma^2_{\mathrm{pred}} \sim \mathcal{LN}(\lambda_{\sigma^2_\mathrm{pred}} = -1, \zeta_{\sigma^2_\mathrm{pred}} = 1)$
%
% Specify these distributions in UQLab separately as two INPUT objects:
SigmaOpts.Marginals(1).Name = 'Sigma2L';
SigmaOpts.Marginals(1).Type = 'Lognormal';
SigmaOpts.Marginals(1).Parameters = [-1 1];

SigmaDist1 = uq_createInput(SigmaOpts);
% 
% SigmaOpts.Marginals(1).Name = 'Sigma2H';
% SigmaOpts.Marginals(1).Type = 'Lognormal';
% SigmaOpts.Marginals(1).Parameters = [-1 1];
% 
% SigmaDist2 = uq_createInput(SigmaOpts);

%%
% Assign these distributions to the discrepancy model options:
DiscrepancyOpts(1).Type = 'Gaussian';
DiscrepancyOpts(1).Prior = SigmaDist1;
% DiscrepancyOpts(2).Type = 'Gaussian';
% DiscrepancyOpts(2).Prior = SigmaDist2;

%% 6 - BAYESIAN ANALYSIS
%
%% 6.1 MCMC solver options
%
% To sample from the posterior distribution, the affine invariant ensemble
% algorithm is chosen, with $400$ iterations and $100$ parallel chains:
Solver.Type = 'MCMC';
Solver.MCMC.Sampler = 'AIES';
Solver.MCMC.Steps = 400;
Solver.MCMC.NChains = 100;

%%
% Enable progress visualization during iteration for the initial prey
% and predator populations (parameters 5 and 6, respectively).
% Update the plots every $40$ iterations:
Solver.MCMC.Visualize.Parameters = 1:nParams;
% Solver.MCMC.Visualize.Interval = 40;

%% 6.2 Posterior sample generation
%
% The options of the Bayesian analysis are gathered within a single
% structure with fields: 
BayesOpts.Type = 'Inversion';
BayesOpts.Name = 'Bayesian model';
BayesOpts.Prior = myPriorDist;
BayesOpts.Data = myData;
BayesOpts.Discrepancy = DiscrepancyOpts;
BayesOpts.Solver = Solver;

%%
% Perform and store in UQLab the Bayesian inversion analysis:
myBayesianAnalysis = uq_createAnalysis(BayesOpts);

%%
% Print out a report of the results:
uq_print(myBayesianAnalysis)

%% 6.3 Posterior sample post-processing

%%
% *Note*: sampling prior predictive samples requires new
% model evaluations

%%

% uq_display(PriorOpts)
% save('PriorOpts')
% close(gcf)
% Display the post processed results:
uq_display(myBayesianAnalysis)
save('myBayesianAnalysis')
close(gcf)
%
% % Diagnose the quality of the results,
% % create a trace plot of the first parameter:
% uq_display(myBayesianAnalysis, 'trace', 1)
% 
% %%
% % From the plots, one can see that several chains have not converged yet.
% % From the trace plot, the non-converged chains are all characterized by a
% % final value $x_1^{(T)}>0.8$:
% badChainsIndex = squeeze(myBayesianAnalysis.Results.Sample(end,1,:) > 0.8);
% 
% %%
% % These chains can be removed from the sample through post-processing. 
% % Additionally, draw a sample of size $10^3$ from the prior and posterior
% % predictive distributions: 
% uq_postProcessInversion(myBayesianAnalysis,...
%                         'badChains', badChainsIndex,...
%                         'prior', 1000,...
%                         'priorPredictive', 1000,...
%                         'posteriorPredictive', 1000);

