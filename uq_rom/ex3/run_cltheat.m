clear all 
t0 = 1;
y0 = sqrt(3/2);
yp0 = 0;

% 
[t,y] = ode15i(@cltheat2,[1 10],y0,yp0);

% ytrue = sqrt(t.^2 + 0.5);
% plot(t,y,t,ytrue,'o');