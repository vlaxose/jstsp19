%***********NOTE************
% The user does not need to call this function directly.  It is called by
% HUT-AMP.
%
% This function has three main functions:
% 1) Initializes the endmemer estimates based 
% 2) Initializes noise variance
% 3) Initializes NNGM parameters on abundances
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

function [Y, Shat, stateFin] = EndExt(Y, N, optALG)

[M,T] = size(Y);

%Initialize the dictionary
if isnumeric(optALG.EEinit)
    Shat = optALG.EEinit;
else
    %Perform VCA to get initial estimate of spectra
    if strcmp(optALG.EEinit,'VCA')
        Shat = vca(Y, 'Endmembers', N,'verbose','off');
        
        %Perform demeaning
        mn = mean(Y(:));
        Shat = Shat - mn;
        Y = Y - mn;
        
    elseif strcmp(optALG.EEinit,'VCAAVG')
        
        Nr = 9;  %number of averages
        Shat = vca(Y, 'Endmembers', N,'verbose','off');
        Shattmp = 0;

        for i = 1:Nr
            Shat2 = vca(Y, 'Endmembers', N,'verbose','off');
            P = find_perm(Shat,Shat2);   %Correct permutation ambiguities
            Shat2 = Shat2*P;
            Shattmp = Shattmp + Shat2;  %keep track of endmembers
        end

        Shat = (Shat + Shattmp)./(Nr + 1);
        
        %Perform demeaning
        mn = mean(Y(:));
        Shat = Shat - mn;
        Y = Y - mn;
    elseif strcmp(optALG.EEinit, 'FSNMF')
        Shat = PCA_FSNMF(Y,N);
        
        %Perform demeaning
        mn = mean(Y(:));
        Shat = Shat - mn;
        Y = Y - mn;

    %Select columns of Y randomly st they are not highly correlated
    elseif strcmp(optALG.EEinit,'data')
        
            Y = Y - mean(Y(:));
            whichCol = 0;
            drawAttempts = 0;
            Ycond = cond(Y);
            while(whichCol <= N) && (drawAttempts < 50)

                %Init
                atomOrder = randperm(T); %different random order each time
                Shat = zeros(M,N);
                Shat(:,1) = Y(:,atomOrder(1))/norm(Y(:,atomOrder(1)));
                whichCol = 2;
                counter = 2;

                %Try to draw the initial dictionary
                while whichCol <= N && counter <= T

                    %Assign the new column
                    Shat(:,whichCol) = Y(:,atomOrder(counter)) /...
                        norm(Y(:,atomOrder(counter)));

                    %Check inner product and increment if different
                    %also check cond to watch for rank deficient cases
                    if (max(abs(Shat(:,1:(whichCol-1))'*Shat(:,whichCol))) < 1) && ...
                            (cond(Shat(:,1:whichCol)) < 10*Ycond)
                        Shat(:,whichCol) = Y(:,atomOrder(counter));
                        whichCol = whichCol + 1;
                    end

                    %Increment counter
                    counter = counter + 1;
                end

                drawAttempts = drawAttempts + 1;

            end
            disp(['Dictionary draw finished after attempts: '...
                num2str(drawAttempts)])

            %Fill in with random if needed
            if whichCol <= N
                Shat(:,whichCol:end) = randn(size(Shat(:,whichCol:end)))/sqrt(M);
            end
    end
end

%% Initialize noise variance
%Initialize noise variance if user has not done so already 
if ~isfield(optALG,'noise_var')
    optALG.SNRdB = min(100,optALG.SNRdB);	% prevent SNR > 100dB 
    optALG.noise_var = norm(Y,'fro')^2/T/(M*(1+10^(optALG.SNRdB/10)));
elseif (optALG.noise_var == 0)
    warning('Since noise_var=0 can cause numerical problems, we have instead set it to a very small value')
    optALG.noise_var = norm(Y,'fro')^2/T/(M*1e10);
elseif (optALG.noise_var < 0)
    error('noise_var<0 is not allowed')
end
%Define noise variance
stateFin.noise_var = resize(optALG.noise_var,M,T);

%% Initialize the NNGM parameters on abundances to fit uniform pdf on [0,1]
if ~isfield(optALG,'L')
    optALG.L = 3;
end
L = optALG.L;

%Initialize lambda
if isfield(optALG,'lambda')
    stateFin.lambda = optALG.lambda;
else
    stateFin.lambda = 1/N;
    %stateFin.lambda = ...
%         1 - sum((sqrt(T) - sum(abs(Y),2)./sqrt(sum(Y.^2,2)))./(sqrt(T)-1))./M;
end;
stateFin.lambda = repmat(stateFin.lambda,N,T);

%load offline-computed initializations for GM parameters
load('inits.mat','init')

%Fit NNGM parameters on abundances to fit uniform pdf on [0,1]
if isfield(optALG,'active_weights')
    stateFin.active_weights = optALG.active_weights;
else
    stateFin.active_weights = zeros(1,1,L);
    stateFin.active_weights(1,1,:) = init(L).active_weights;
    stateFin.active_weights = repmat(stateFin.active_weights, [N T 1]);
end

if isfield(optALG,'active_loc')
    stateFin.active_loc = optALG.active_loc;
else
    stateFin.active_loc = zeros(1,1,L);
    %Shift to "positive" [0, 1] uniform prior
    stateFin.active_loc(1,1,:) = init(L).active_mean+0.5;
    stateFin.active_loc = repmat(stateFin.active_loc, [N T 1]);
end
if isfield(optALG,'active_scales')
    stateFin.active_scales = optALG.active_scales;
else
    stateFin.active_scales = zeros(1,1,L);
    stateFin.active_scales(1,1,:) = init(L).active_var;
    stateFin.active_scales = repmat(stateFin.active_scales, [N T 1]);
end

return