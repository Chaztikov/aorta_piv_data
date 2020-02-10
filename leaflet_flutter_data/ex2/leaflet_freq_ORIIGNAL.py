close all
clear
clf
%Edit Filename
%data_directory = '/home/chaztikov/git/aorta_piv_data/data/original/'
%fileID = fopen(data_directory+'OpenAreaPerimountWaterbpm60.txt');

fileID = fopen('/home/chaztikov/git/aorta_piv_data/data/original/OpenAreaPerimountWaterbpm60.txt')

A = textscan(fileID,'%f %f','Delimiter',',');
fclose(fileID)

%Edit BPM
BPM = 80;
InitialFrame =  000;
StartFFTFrame = 1;
NFFT = 512*4;
FPS = 5000;
PixelsPerCm = 100;


Area = A{1,2}/(PixelsPerCm^2);
Time =(1:FPS)/FPS*1000;
FramesPerPulse = 60/BPM*FPS;
Pulses = floor((size(Area,1)-InitialFrame)/FramesPerPulse);

clf
figure(1)
title('Open Area vs. Frame')
xlabel('Frame')
ylabel('Valve Open Area [cm^2]')
set(gca,'FontSize',20)

hold on

for i=1:Pulses
    for j=1:FramesPerPulse
        PulseData(i,j) = Area((i-1)*FramesPerPulse+j+InitialFrame);
    end
    plot(PulseData(i,:))
    drawnow
end



%perform FFT
PulseFreq = FPS*(0:(NFFT/2))/NFFT;
figure(2)
hold on
title('Valve Area Frequency')
xlabel('Frequency [Hz]')
ylabel('|Power|')
set(gca,'FontSize',20)
xlim([0 50])
for i=1:Pulses
    PulseDataFFT(i,:) = fft(PulseData(i,StartFFTFrame:(StartFFTFrame+NFFT-1)));
    Power = abs(PulseDataFFT(i,:)/NFFT);
    plot(PulseFreq,Power(1:NFFT/2+1))
end
ylim([0 1])


%Find peak
figure(3)
hold on
title('Fluttering Frequency (first peak)')
xlabel('Frequency [Hz]')
ylabel('Number of Pulses')
set(gca,'FontSize',20)
Peaks=zeros(size(PulseFreq,2),1);
for i=1:Pulses
    Power = abs(PulseDataFFT(i,:)/NFFT);
    j=1;
    while Power(j)-Power(j+1)>0 || Power(j+2)-Power(j+1)>0
        j=j+1;
    end
    Peaks(j+1) = Peaks(j+1)+1;
end
bar(PulseFreq,Peaks)
xlim([0 30])

filename = sprintf('FlutteringFrequency%dbpm.png',BPM);
saveas(gcf,filename)

%difference from smoothed
figure(4)
hold on
title('Fluttering Frequency (Difference from smoothed)')
xlabel('Frequency [Hz]')
ylabel('Number of Pulses')
set(gca,'FontSize',20)
Peaks=zeros(size(PulseFreq,2),1);
for i=1:Pulses
    PulseDataSmoothed = smooth(PulseData(i,:),150);
    Variance = PulseData(i,:)-PulseDataSmoothed';
    VarianceFFT(i,:) = fft(Variance(StartFFTFrame:(StartFFTFrame+NFFT-1)));
    Power = abs(VarianceFFT(i,:)/NFFT);
    plot(PulseFreq,Power(1:NFFT/2+1))
end
xlim([0 50])
filename = sprintf('VarianceFrequency%dbpm.png',BPM);
saveas(gcf,filename)