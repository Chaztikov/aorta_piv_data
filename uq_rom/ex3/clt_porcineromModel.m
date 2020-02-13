function population = uq_porcineromModel(X,time)
% implementation of the Lotka-Volterra predator-prey model with multiple
% parameter realizations and the time

firstTime = time(1);
lastTime = time(end);    % Duration time of simulation.

nSteps = numel(time); % Number of timesteps
%% Initialize
nReal = size(X,1); 

% Parameters
R1 = X(:,1);
R2 =  X(:,2);
C = X(:,3);
P0 = X(:,4);

% Initial conditions
initialP1 = X(:,5);

% Measured Function Values
Pin = X(:,6);
Pout = X(:,7);
Q =  X(:,8);




%% Solve equation

% solver options (smaller tolerance)
odeOpts = odeset('RelTol',1e-4,'AbsTol',1e-7);
%odeOpts = odeset('RelTol',2e-3,'AbsTol',2e-6);

% for loop to solve equations with multiple initial values and parameters
population = zeros(nReal,2*nSteps);
for ii = 1:nReal
    
    %set up kernel
    %kernel = @(t,ii)( epx( ( 1/R1(ii) + 1/R2(ii) ) / C(ii) * (t - 0) ))
    
    % setup diff equations 
    diffEq=@(t,x) [ -(1 + R1(ii) / R2(ii) ) * x(1) + ( R1(ii) / R2(ii) ) * Pin(ii) + P0(ii) / (R1(ii)*C(ii)) ]
    
    %[  x(1)*(alpha(ii) - beta(ii)*x(2));...
    %-x(2)*(gamma(ii) - delta(ii)*x(1))];
    
    % solve using numerical ODE solver 45
    [t,sol] = ode45(diffEq,[firstTime lastTime],[initialP1(ii)], odeOpts);
    
    % interpolate solution to specified timesteps
    interpSolP1 = interp1(t,sol(:,1),time);
    
    % assign solution
    population(ii,:) = [interpSolP1'];

end

