
function [yv, Phi] = Adaptive_Channel_Estimation_Multi_Path2()
%------------------------System Parameters---------------------------------
Num_BS_Antennas=64; % BS antennas
BSAntennas_Index=0:1:Num_BS_Antennas-1; % Indices of the BS Antennas
Num_BS_RFchains=16; % BS RF chains

Num_MS_Antennas=32; % MS antennas
MSAntennas_Index=0:1:Num_MS_Antennas-1; % Indices of the MS Antennas
Num_MS_RFchains=8;  % MS RF chains

Num_Qbits=7;  % Number of phase shifters quantization bits

% % ---------------------Channel Parameters ---------------------------------
Num_paths=3; % Number of channel paths

%path-loss calculation
Carrier_Freq=28*10^9; % Carrier frequency
lambda=3*10^8/Carrier_Freq; % Wavelength
n_pathloss=3; % Pathloss constant
Tx_Rx_dist=50; % Distance between BS and MS
ro=((lambda/(4*pi*5))^2)*(5/Tx_Rx_dist)^n_pathloss; % Pathloss
Pt_avg=10^(.7); % Average total transmitted power
Pr_avg=Pt_avg*ro; % Average received power

% Noise calculations
Bandwidth=100*10^6;   % Channel bandwidth
No_dB=-173+10*log10(Bandwidth); % Noise power in dB
No=10^(.1*No_dB);    % Noise power absolute

%---------------- Channel Estimation Algorithm Parameters------------------
G_BS=192; % Required resolution for BS AoD
G_MS=192; % Required resolution for MS AoA

K_BS=2;  % Number of Beamforming vectors per stage 
K_MS=2;  % Number of Measurement vectors per stage

Num_paths_est=Num_paths; % Number of desired estimated paths  

S=floor(max(log(G_BS/Num_paths_est)/log(K_BS),log(G_MS/Num_paths_est)/log(K_MS))); % Number of iterations

% Optimized power allocation
Pr_alloc=power_allocation(Num_BS_Antennas,Num_BS_RFchains,BSAntennas_Index,G_BS,G_MS,K_BS,K_MS,Num_paths_est,Num_Qbits);
Pr=abs(Pr_avg*Pr_alloc*S);

%---------------------- Simulation Parameters-------------------------------
ITER=100; % Number of independent realizations (to be averaged over)
SNR_dBa=-40:5:0; % SNR range in dB (for beamforming - after channel estimation)

%Rate_PerfectCSI=zeros(1,length(SNR_dBa)); % Achievable rate with perfect channel knowledge
%Rate_EstimatedCSI=zeros(1,length(SNR_dBa)); % Achievable rate with estimated channel

% Beamsteering vectors generation
for g=1:1:G_BS
    AbG(:,g)=sqrt(1/Num_BS_Antennas)*exp(1j*(2*pi)*BSAntennas_Index*((g-1)/G_BS));
end

% Am generation
for g=1:1:G_MS
    AmG(:,g)=sqrt(1/Num_MS_Antennas)*exp(1j*(2*pi)*MSAntennas_Index*((g-1)/G_MS));
end
%--------------------------------------------------------------------------

for iter=1:1:ITER
    iter 
% Channel Generation 

% Channel parameters (angles of arrival and departure and path gains)
AoD=2*pi*rand(1,Num_paths);
AoA=2*pi*rand(1,Num_paths);
alpha=(sqrt(1/2)*sqrt(1/Num_paths)*(randn(1,Num_paths)+1j*randn(1,Num_paths)));

% Channel construction
Channel=zeros(Num_MS_Antennas,Num_BS_Antennas);
for l=1:1:Num_paths
Abh(:,l)=sqrt(1/Num_BS_Antennas)*exp(1j*BSAntennas_Index*AoD(l));
Amh(:,l)=sqrt(1/Num_MS_Antennas)*exp(1j*MSAntennas_Index*AoA(l));
Channel=Channel+sqrt(Num_BS_Antennas*Num_MS_Antennas)*alpha(l)*Amh(:,l)*Abh(:,l)';
end

%Algorithm parameters initialization
KB_final=[]; % To keep the indecis of the estimated AoDs
KM_final=[]; % To keep the indecis of the estimated AoAs
yv_for_path_estimation=zeros(K_BS*K_MS*Num_paths_est^2,1); % To keep received vectors

for l=1:1:Num_paths_est % An iterations for each path
KB_star=1:1:K_BS*Num_paths_est; % Best AoD ranges for the next stage
KM_star=1:1:K_MS*Num_paths_est; % Best AoA ranges for the next stage

for t=1:1:S
    
%Generating the "G" matrix in the paper used to construct the ideal
%training beamforming and combining matrices - These matrices capture the
%desired projection of the ideal beamforming/combining vectors on the
%quantized steering directions

%BS G matrix
G_matrix_BS=zeros(K_BS*Num_paths_est,G_BS);
Block_size_BS=G_BS/(Num_paths_est*K_BS^t);
Block_BS=[ones(1,Block_size_BS)];
for k=1:1:K_BS*Num_paths_est
    G_matrix_BS(k,(KB_star(k)-1)*Block_size_BS+1:(KB_star(k))*Block_size_BS)=Block_BS;
end

%MS G matrix generation
G_matrix_MS=zeros(K_MS*Num_paths_est,G_MS);
Block_size_MS=G_MS/(Num_paths_est*K_MS^t);
Block_MS=[ones(1,Block_size_MS)];
for k=1:1:K_MS*Num_paths_est
    G_matrix_MS(k,(KM_star(k)-1)*Block_size_MS+1:(KM_star(k))*Block_size_MS)=Block_MS;
end

% Ideal vectors generation
F_UC=(AbG*AbG')^(-1)*(AbG)*G_matrix_BS';
W_UC=(AmG*AmG')^(-1)*(AmG)*G_matrix_MS';

% Ideal vectors normalization
F_UC=F_UC*diag(1./sqrt(diag(F_UC'*F_UC)));
W_UC=W_UC*diag(1./sqrt(diag(W_UC'*W_UC)));

% Hybrid Precoding Approximation
for m=1:1:K_BS*Num_paths_est
[F_HP(:,m)]=HybridPrecoding(F_UC(:,m),Num_BS_Antennas,Num_BS_RFchains,Num_Qbits);
end
for n=1:1:K_MS*Num_paths_est
[W_HP(:,n)]=HybridPrecoding(W_UC(:,n),Num_MS_Antennas,Num_MS_RFchains,Num_Qbits);
end
  
% Noise calculations
Noise=W_HP'*(sqrt(No/2)*(randn(Num_MS_Antennas,K_BS*Num_paths_est)+1j*randn(Num_MS_Antennas,K_BS*Num_paths_est)));

% Received signal
Y=sqrt(Pr(t))*W_HP'*Channel*F_HP+Noise;
yv=reshape(Y,K_BS*K_MS*Num_paths_est^2,1); % vectorized received signal
if(t==S)
    yv_for_path_estimation=yv_for_path_estimation+yv/sqrt(Pr(t));
end

A1=transpose(F_HP)*conj(AbG);
     A2=W_HP'*AmG;
     E=kron(A1,A2);
     
     Phi = sqrt(Pr(t))*E;



% Subtracting the contribution of previously estimated paths
%for i=1:1:length(KB_final)
%     A1=transpose(F_HP)*conj(AbG(:,KB_final(i)+1));
%     A2=W_HP'*AmG(:,KM_final(i)+1);
%     Prev_path_cont=kron(A1,A2);
%     Alp=Prev_path_cont'*yv;
%    yv=yv-Alp*Prev_path_cont/(Prev_path_cont'*Prev_path_cont);
%end 
 
% Maximum power angles estimation
%Y=reshape(yv,K_MS*Num_paths_est,K_BS*Num_paths_est); 
%[val mX]=max(abs(Y));
%Max=max(val);
%[KM_temp KB_temp]=find(abs(Y)==Max);
%KM_max(1)=KM_temp(1);
%KB_max(1)=KB_temp(1);

% Keeping the best angle in a history matrix (the T matrix in Algorithm 3)
%KB_hist(l,t)=KB_star(KB_max(1));
%KM_hist(l,t)=KM_star(KM_max(1));

% Final AoAs/AoDs
%if(t==S)
 %   KB_final=[KB_final KB_star(KB_max(1))-1];
  %  KM_final=[KM_final KM_star(KM_max(1))-1];
  %  W_paths(l,:,:)=W_HP;
   % F_paths(l,:,:)=F_HP;
%end
  

%TempB=KB_star;
%TempM=KM_star;

% Adjusting the directions of the next stage (The adaptive search)
%for ln=1:1:l 
%KB_star((ln-1)*K_BS+1:ln*K_BS)=(KB_hist(ln,t)-1)*K_BS+1:1:(KB_hist(ln,t))*K_BS;
%KM_star((ln-1)*K_MS+1:ln*K_MS)=(KM_hist(ln,t)-1)*K_MS+1:1:(KM_hist(ln,t))*K_MS;
%end

end % -- End of estimating the lth path

end %--- End of estimation of the channel


% Estimated angles
%AoD_est=2*pi*KB_final/G_BS;
%AoA_est=2*pi*KM_final/G_MS;

% Estimated paths 
%Wx=zeros(Num_MS_Antennas,K_MS*Num_paths_est);
%Fx=zeros(Num_BS_Antennas,K_BS*Num_paths_est);
%Epsix=zeros(K_BS*K_MS*Num_paths_est^2,Num_paths_est);

%for l=1:1:Num_paths_est
%Epsi=[];
%Wx(:,:)=W_paths(l,:,:);
%Fx(:,:)=F_paths(l,:,:);
%for i=1:1:length(KB_final)
     %A1=transpose(Fx)*conj(AbG(:,KB_final(i)+1));
     %A2=Wx'*AmG(:,KM_final(i)+1);
     %E=kron(A1,A2);
    % Epsi=[Epsi E];
%end 
%Epsix=Epsix+Epsi;
%end
%alpha_est=Epsix\yv_for_path_estimation;

% Reconstructing the estimated channel
%Channel_est=zeros(Num_MS_Antennas,Num_BS_Antennas);
%for l=1:1:Num_paths_est
%Abh_est(:,l)=sqrt(1/Num_BS_Antennas)*exp(1j*BSAntennas_Index*AoD_est(l));
%Amh_est(:,l)=sqrt(1/Num_MS_Antennas)*exp(1j*MSAntennas_Index*AoA_est(l));
%Channel_est=Channel_est+alpha_est(l)*Amh_est(:,l)*Abh_est(:,l)';    
%end

% Optimal SVD precoders
%[U_H S_H V_H]=svd(Channel);
%F_opt=V_H(:,1:Num_paths_est);
%W_opt=U_H(:,1:Num_paths_est);

% Hybrid Precoding for Data transmission based on the estimated channel
%[U_H_est S_H_est V_H_est]=svd(Channel_est);
%F_opt_est=V_H_est(:,1:Num_paths_est);
%W_opt_est=U_H_est(:,1:Num_paths_est);

%for m=1:1:K_BS*Num_paths_est
%[F_HP_data]=HybridPrecoding(F_opt_est,Num_BS_Antennas,Num_BS_RFchains,Num_Qbits);
%end
%for n=1:1:K_MS*Num_paths_est
%[W_HP_data]=HybridPrecoding(W_opt_est,Num_MS_Antennas,Num_MS_RFchains,Num_Qbits);
%end

% Spectral efficiency calculations
%count=0;
%for SNR_dB=SNR_dBa
 %   count=count+1;
  %  SNR=10^(.1*SNR_dB);
   % No=Pr_avg/SNR;
    
%G_opt=sqrt(Pr_avg/Num_paths_est)*W_opt'*Channel*F_opt;
%G_est=sqrt(Pr_avg/Num_paths_est)*W_HP_data'*Channel*F_HP_data;

%Rate_PerfectCSI(count)=Rate_PerfectCSI(count)+abs(log2(det(eye(Num_paths_est)+((No)^-1)*(G_opt*G_opt'))));
%Rate_EstimatedCSI(count)=Rate_EstimatedCSI(count)+abs(log2(det(eye(Num_paths_est)+((No)^-1)*(G_est*G_est'))));
%end

end 

% Rate averaging
%Rate_PerfectCSI=Rate_PerfectCSI/ITER;
%Rate_EstimatedCSI=Rate_EstimatedCSI/ITER;

% Plotting the rates
%plot(SNR_dBa,Rate_PerfectCSI,'r','LineWidth',1.5);
%hold on; plot(SNR_dBa,Rate_EstimatedCSI,'b','LineWidth',1.5);
%legend('Unconstrained Precoding - Perfect CSI','Hybrid Precoding - Estimated CSI')
%xlabel('SNR (dB)','FontSize',12);
%ylabel('Spectral Efficiency (bps/ Hz)','FontSize',12);