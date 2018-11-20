function [ U_hat, R,Outliers,time_data,estHist ] = grasta_mc_timing( I,J,S, numr,numc,maxCycles,CONVERGE_LEVLE,OPTIONS,error_function)
%  GRASTA (Grassmannian Robust Adaptive Subspace Tracking Algorithm) robust
%  matrix completion code
%  by Jun He and Laura Balzano, Sept. 2011.
%
%   Online Robust Subspace Tracking from Partial Information
%       http://arxiv.org/abs/1109.3827
%
% Inputs:
%
%       (I,J,S) index the known entries across the entire data set X. So we
%       know that for all k, the true value of X(I(k),J(k)) = S(k)
%
%       numr = number of rows
%       numc = number of columns
%           NOTE: you should make sure that numr<numc.  Otherwise, use the
%           transpose of X
%
% Outputs:
%       U and R such that UR' approximates X.
%
%
% Form some sparse matrices for easier matlab indexing
values    = sparse(I,J,S,numr,numc);
Indicator = sparse(I,J,1,numr,numc);

status = struct();  % maintain GRASTA running status
status.init  = 0;   % will be set 1 once GRASTA start working

OPTS  = struct(); % initial a empty struct for OPTS
U_hat = zeros(1); % U_hat will be initialized in GRASTA

%Preallocate
estHist.errZ = zeros(maxCycles,1);
time_data = zeros(maxCycles,1);
step1_times = zeros(maxCycles,1);
step2_times = zeros(maxCycles,1);

Zold = 0;
for outIter = 1 : maxCycles,
    
    %Start timing
    tstart = tic;
    
    col_order = randperm(numc);
    for k = 1 : numc,
        idx = find(Indicator(:,col_order(k)));
        v_Omega = values(idx,col_order(k));
        
        if length(idx) < OPTIONS.RANK * 1,
            continue;
        end
        
        [U_hat, status, OPTS] = grasta_stream(v_Omega, idx, U_hat, status, OPTIONS, OPTS);
        
    end
    
    
    
    %Stop timing of step 1
    step1_times(outIter) = toc(tstart);
    
    %Do step 2 every time. We are timing them separately to make the
    %comparison fair, as step 2 only actually gets done on the LAST
    %iteration.
    %Start timing
    tstart = tic;
    
    % OPTS2 used for recovering R
    OPTS2 = OPTS;
    
    R = zeros(numc,OPTIONS.RANK);
    Outliers =zeros(numc,numr);
    for k=1:numc,
        idx = find(Indicator(:,k));
        v_Omega = values(idx,k);
        
        if length(idx) < OPTIONS.RANK * 1,
            continue;
        end
        
        U_Omega = U_hat(idx,:);
        
        if OPTIONS.USE_MEX,
            [s, w, ~] = mex_srp(U_Omega, v_Omega, OPTS2);
        else
            [s, w, ~] = admm_srp(U_Omega, v_Omega, OPTS2);
        end
        
        R(k,:) = w';
        Outliers(k,idx) = s;
    end
    
    %Stop timing of step 2
    step2_times(outIter) = toc(tstart);
    
    %Compute timing for this iteration and save current estimate. Again,
    %notice that we are not timing this part
    
    %Time for this iteration is the sum of the step 1 times, plus only the
    %most recent step 2 time
    time_data(outIter) = sum(step1_times(1:outIter)) + step2_times(outIter);
    
    %Save the result
    Zhat = U_hat*R';
    estHist.errZ(outIter) = error_function(Zhat);
    
    %Check for convergence
    stopCriterion = norm(Zhat - Zold,'fro') /...
        norm(Zhat,'fro');
    if outIter > 10
        if stopCriterion < OPTIONS.stopTol
            break;
        end
    end
    Zold = Zhat;
    
    if status.level >= CONVERGE_LEVLE,
        %         fprintf('Pass %d/%d, reach the convergence level - %d...\n',outIter, maxCycles,status.level);
        break;
    end
    
    %     fprintf('Pass %d/%d ......\n',outIter, maxCycles);
end

%Trim the results if all iterations were not used
time_data = time_data(1:outIter);
estHist.errZ = estHist.errZ(1:outIter);



