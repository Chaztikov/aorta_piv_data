close all
clear all

%fileID = fopen('/home/chaztikov/git/aorta_piv_Data/Data/original/OpenAreaPerimountWaterbpm60.txt')

% close all
% clear
% clf
% %Edit Filename
% 
% %data_directory = '/home/chaztikov/git/aorta_piv_data/data/original/'
% %fileID = fopen(data_directory+'OpenAreaPerimountWaterbpm60.txt');
% 
% fileID = fopen('/home/chaztikov/git/aorta_piv_data/data/original/OpenAreaPerimountWaterbpm60.txt')
% 
% A = textscan(fileID,'%f %f','Delimiter',',');
% fclose(fileID)
% 
% %Edit BPM
% BPM = 60;
% InitialFrame = 12000;
% StartFFTFrame = 2000;
% NFFT = 512*4;
% FPS = 5000;
% PixelsPerCm = 100;
% 
% 
% Area = A{1,2}/(PixelsPerCm^2);
% Time =(1:FPS)/FPS*1000;
% FramesPerPulse = 60/BPM*FPS;
% Pulses = floor((size(Area,1)-InitialFrame)/FramesPerPulse);
% 
% 

fileID = fopen('/home/chaztikov/git/aorta_piv_data/data/aortic_valve_piv_data.csv')
Data = textscan(fileID,'%f %f %f %f %f','Delimiter',',');
% [tstep Pin Q Pout pdva] = textscan(fileID,'%f %f %f %f %f','Delimiter',',');
% [tstep,Pin, Q, Pout, pdva]  = Data{1}
% clear Data

tstep=Data{1} *0.03333/1000
Pin=Data{2}
Q=Data{3}
Pout=Data{4}
pdva=Data{5}
clear Data

fclose(fileID)

save('uq_porcineromModel.mat')
