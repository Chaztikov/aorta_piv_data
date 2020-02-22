function population = uq_porcinerom_downstream_Model(X,time)
% implementation of the 
% Reduced Order Model for the upstream aorta model
% model with multiple
% parameter realizations and the time


firstTime = time(1);
lastTime = time(end);    % Duration time of simulation.

nSteps = numel(time); % Number of timesteps
%% Initialize
nReal = size(X,1); 

nparam=0;
% Parameters

nparam=nparam+1;
R1 = X(:,nparam);

nparam=nparam+1;
R2 =  X(:,nparam);

nparam=nparam+1;
C = X(:,nparam);

% nparam=nparam+1
% P0 = X(:,nparam);

% nparam=nparam+1
% initialP1 = X(:,nparam);

% nparam=nparam+1
% Pin = X(:,6);

nparam=nparam+1;
Pout = X(:,nparam);

% nparam=nparam+1
% Q =  X(:,8);




%% Solve equation

% solver options (smaller tolerance)
odeOpts = odeset('RelTol',1e-4,'AbsTol',1e-7);
%odeOpts = odeset('RelTol',2e-3,'AbsTol',2e-6);

% for loop to solve equations with multiple initial values and parameters
nVars = 1;

population = zeros(nReal, nVars * nSteps);
for ii = 1:nReal
    
    %set up kernel
    %kernel = @(t,ii)( epx( ( 1/R1(ii) + 1/R2(ii) ) / C(ii) * (t - 0) ))
    
    % setup diff equations 
%     diffEq=@(t,x)  ([x(1); ...
%                     x(1)])
%     diffEq=@(t,x)  ([x(1)]);

%     diffEq=@(t,x)  -(1 + R1(ii) ./ R2(ii) ) .* x + ( R1(ii) ./ R2(ii) ) .* Pin(ii) + P0(ii) ./ (R1(ii).*C(ii)) 
%     diffEq=@(t,x) [ -(1 + R1(ii) / R2(ii) ) * x(1) + ( R1(ii) / R2(ii) ) * Pin(ii) + P0(ii) / (R1(ii)*C(ii)) ];
    diffEq=@(t,x) [ -(1 + R1(ii) / R2(ii) ) * x(1) + ( R1(ii) / R2(ii) ) * Pout(ii)  / (R1(ii)*C(ii)) ];
    
    %[  x(1)*(alpha(ii) - beta(ii)*x(2));...
    %-x(2)*(gamma(ii) - delta(ii)*x(1))];
    
%     [t,sol] = ode45(diffEq,[firstTime lastTime],[initialP1(ii)], odeOpts);
%     [t,sol] = ode45(diffEq,[firstTime lastTime],[Pout(1)], odeOpts);
    [t,sol] = ode45(diffEq,[firstTime lastTime],[0], odeOpts);
    
    % interpolate solution to specified timesteps
    interpSolP1 = interp1(t,sol(:,1),time);
    
    % assign solution
    population(ii,:) = [interpSolP1'];

end

