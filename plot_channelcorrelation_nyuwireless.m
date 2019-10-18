clear;
clc;

addpath('basic_system_functions');
addpath(genpath('benchmark_algorithms'));

Nt = 4;
Nr = 32;
Gr = Nr;
Gt = Nt;
total_num_of_clusters = 2;
total_num_of_rays = 3;
Np = total_num_of_clusters*total_num_of_rays;
L = 4;

[H_sim,Zbar,Ar,At,Dr,Dt] = wideband_mmwave_channel(L, Nr, Nt, total_num_of_clusters, total_num_of_rays, Gr, Gt);
load('./basic_system_functions/nywireless_channel.mat');
H_nyw = Hf{200};
Z1 = Dr'*H_sim(:,:,1)*Dt;
Z2 = Dr'*H_nyw(:, 1:Nt)*Dt;

C1 = Z1*Z1';
C2 = Z2*Z2';
figure;
subplot(1,2,1)
surf(abs(C1/norm(C1)))
xlabel('N_R')
ylabel('N_R')
zlabel('Amplitude of the channel correlation')


subplot(1,2,2)
surf(abs(C2/norm(C2)))
xlabel('N_R')
ylabel('N_R')
zlabel('Amplitude of the channel correlation')