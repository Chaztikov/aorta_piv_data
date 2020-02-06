c = pca(PulseData')
figures
plot(PulseData'*c(:,1))
figure
plot(PulseData'*c(:,2))