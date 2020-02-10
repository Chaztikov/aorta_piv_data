close all
clear

%fileID = fopen('/home/chaztikov/git/aorta_piv_Data/Data/original/OpenAreaPerimountWaterbpm60.txt')

clear all
fileID = fopen('/home/chaztikov/git/aorta_piv_data/data/aortic_valve_piv_data.csv')
Data = textscan(fileID,'%f %f %f %f %f','Delimiter',',');


tstep=transpose(Data{1})
Pin=transpose(Data{2})
Q=transpose(Data{3})
Pout=transpose(Data{4})
pdva=transpose(Data{5})

fclose(fileID)

save('uq_porcineromModel.mat')
