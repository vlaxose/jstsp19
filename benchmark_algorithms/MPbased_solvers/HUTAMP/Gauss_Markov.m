% Gauss Markov process
%
% This function accounts for amplitude correlations by assuming a Gauss 
% Markov process.  Given a set of means (eta) and variances (kappa) of the
% incoming messages, this function calculates the according outgoing
% ----------NOTE-----------
% This function is to be used internally by HUTAMP
%
% Inputs
% - param
% -- alpha      This parameters captures the correlations among the
%               elements.  Must take value between 0 and 1.
% -- mean       Overall mean of the process.
% -- var        Overall variance of the process.
% --KAPPA_OUT   Variances of the messages from BiGAMP
% --ETA_OUT     Means of the messages from BiGAMP
%
% Outputs   The same as the inputs except computes outgoing messages from
%           the Gauss Markov Process.
%
% Coded by: Justin Ziniel, Jeremy Vila The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu, vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%       - Created (12/10/11; JAZ)
%       - Added genRand method (01/03/12; JAZ)
%       - Implemented EM learning of THETA, PHI, and alpha (01/13/12; JAZ)
% Version 1.0

function [ETA_IN, KAPPA_IN, param] ...
    = Gauss_Markov(ETA_OUT, KAPPA_OUT, param, optALG)

% Begin by getting size information, and verifying validity of
% the THETA_OUT and KAPPA_OUT inputs
[M, N] = size(ETA_OUT);      % # of Gaussian components: L

% Now we need to compute the perturbation process variance RHO,
% which is set based on the values of alpha and PHI to ensure
% the process has a steady-state variance of PHI
RHO = (2 - param.specCorr) .* param.specVar ./ param.specCorr;

% Initialize certain initial and terminal messages
THETA_FWD = NaN([M, N]);
PHI_FWD = NaN([M, N]);
THETA_BWD = NaN([M, N]);
PHI_BWD = NaN([M, N]);

THETA_FWD(1,:,:) = param.specMean(1,:);
PHI_FWD(1,:,:) = param.specVar(1,:);
THETA_BWD(M,:,:) = zeros([1, N]);
PHI_BWD(M,:,:) = inf([1, N]);

% First execute the forward portion of the forward/backward
% pass.  Need to differentiate whether the Gauss-Markov process
% is defined for rows or columns of the amplitude matrix
% (or matrices, for D > 1)

for m = 1:M-1
    TempVar = (1./PHI_FWD(m,:) + 1./KAPPA_OUT(m,:)).^(-1);
    THETA_FWD(m+1,:) = (1 - param.specCorr(m+1,:)) .* ...
        TempVar .* (THETA_FWD(m,:)./PHI_FWD(m,:) + ...
        ETA_OUT(m,:)./KAPPA_OUT(m,:)) + ...
        param.specCorr(m+1,:) .* param.specMean(m+1,:);
    PHI_FWD(m+1,:) = (1 - param.specCorr(m+1,:)).^2 .* ...
        TempVar + param.specCorr(m+1,:).^2 .* RHO(m+1,:);
end

% Now execute the backward portion of the forward/backward pass
for m = M:-1:2
    TempVar = (1./PHI_BWD(m,:) + 1./KAPPA_OUT(m,:)).^(-1);
    THETA_BWD(m-1,:) = 1./(1 - param.specCorr(m-1,:)) .* ...
        (TempVar .* (THETA_BWD(m,:)./PHI_BWD(m,:) + ...
        ETA_OUT(m,:)./KAPPA_OUT(m,:)) - ...
        param.specCorr(m-1,:) .* param.specMean(m-1,:));
    PHI_BWD(m-1,:) = (1 - param.specCorr(m-1,:)).^2 .* ...
        (TempVar + param.specCorr(m-1,:).^2 .* RHO(m-1,:));
end

% Now the forward/backward pass is complete.  Combine the
% forward and backward messages to yield the outgoing messages
KAPPA_IN = (PHI_FWD.^(-1) + PHI_BWD.^(-1)).^(-1);
ETA_IN = (THETA_FWD./PHI_FWD + THETA_BWD./PHI_BWD);
ETA_IN = KAPPA_IN .* ETA_IN;

% Compute the posterior means and variances as well
POST_VAR = (1./KAPPA_OUT + 1./PHI_FWD + 1./PHI_BWD).^(-1);
POST_MEAN = (ETA_OUT ./ KAPPA_OUT + THETA_FWD ./ PHI_FWD + ...
    THETA_BWD ./ PHI_BWD);
POST_MEAN = POST_VAR .* POST_MEAN;

%Learn Gaus Markov parameters
if optALG.learn_GausMark
% ------------------------------------
%   Updated amplitude variance, sigma^2
% ------------------------------------

% For now, we don't support groups, but keep the group
% indexing for future use.  Also check to make sure
% that the user's parameter learning preferences make
% sense
g_ind = 1:N;    % Group indices (all)
N_g = numel(g_ind);
AHr = 2:M;      % Ahead row index
BHr = 1:M-1;    % Behind row index
AHc = g_ind;   	% Ahead column index
BHc = g_ind;    % Behind column index

% Start by computing E[GAMMA(n,t)'*GAMMA(n,t-1)|Y] (or 
% E[GAMMA(n,t)'*GAMMA(n-1,t)|Y])
Q = (1./KAPPA_OUT(AHr,AHc) + 1./PHI_BWD(AHr,AHc)).^(-1);
R = (ETA_OUT(AHr,AHc)./KAPPA_OUT(AHr,AHc) + ...
    THETA_BWD(AHr,AHc)./PHI_BWD(AHr,AHc));
Q_BAR = (1./KAPPA_OUT(BHr,BHc) + 1./PHI_FWD(BHr,BHc)).^(-1);     
R_BAR = (ETA_OUT(BHr,BHc)./KAPPA_OUT(BHr,BHc) + ...
    THETA_FWD(BHr,BHc)./PHI_FWD(BHr,BHc));        
Q_TIL = (1./Q_BAR + ((1-param.specCorr(BHr,BHc)).^2)./(Q + ...
    (param.specCorr(BHr,BHc).^2 .* RHO(BHr,BHc)))).^(-1);        
M_BAR = (1 - param.specCorr(BHr,BHc)).*(Q.*R - ...
    (param.specCorr(BHr,BHc).*param.specMean(BHr,BHc))) ./ (Q + ...
    (param.specCorr(BHr,BHc).^2 .* RHO(BHr,BHc))) + R_BAR;        
GAMMA_CORR = (Q./(Q + (param.specCorr(BHr,BHc).^2.*RHO(BHr,BHc)))) .* ...
    ((1-param.specCorr(BHr,BHc)).*(Q_TIL + abs(Q_TIL.*M_BAR).^2) + ...
    (param.specCorr(BHr,BHc).*param.specMean(BHr,BHc)).*Q_TIL.*conj(M_BAR) + ...
    (param.specCorr(BHr,BHc).^2.*RHO(BHr,BHc)).*Q_TIL.*conj(M_BAR).*R);

% Now compute E[|GAMMA(n,t) - (1-alpha)*GAMMA(n,t-1) -
% alpha*THETA|^2 | Y] (or similar for column GM process)
E1 = POST_VAR(AHr,AHc) + abs(POST_MEAN(AHr,AHc)).^2 - ...
    2*(1 - param.specCorr(BHr,BHc)) .* real(GAMMA_CORR) - ...
    2*param.specCorr(BHr,BHc) .* real(conj(param.specMean(AHr,AHc)) .* ...
    POST_MEAN(AHr,AHc)) + (1 - param.specCorr(BHr,BHc)).^2 .* ...
    (POST_VAR(BHr,BHc) + abs(POST_MEAN(BHr,BHc)).^2) + ...
    2*param.specCorr(BHr,BHc) .* (1 - param.specCorr(BHr,BHc)) .* ...
    real(conj(param.specMean(BHr,BHc)) .* POST_MEAN(BHr,BHc)) + ...
    param.specCorr(BHr,BHc).^2 .* abs(param.specMean(BHr,BHc)).^2;

% Now compute E[|GAMMA(n,0) - THETA|^2 | Y], (or
% similar for column GM process)
E2 = POST_VAR(1,g_ind) + ...
    abs(POST_MEAN(1,g_ind)).^2 - ...
    2*real(conj(param.specMean(1,g_ind)) .* ...
    POST_MEAN(1,g_ind)) + abs(param.specMean(1,g_ind)).^2;

sig_upd = 1 ./ (M*param.specCorr(1,:).* ...
            (2 - param.specCorr(1,:))) .* ...
            sum(E1, 1) + (1/M)*E2;
% sig_upd = 1 ./ ((M-1)*param.specCorr(1,:).^2).* ...
%             sum(E1, 1) + E2/M;
        
RHO = (2 - param.specCorr) .* param.specVar ./ param.specCorr;

% ------------------------------------
%   Updated amplitude mean, kappa
% ------------------------------------
% First compute E[GAMMA(n,t) - (1-alpha)*THETA(n,t-1) |
% Y] * (1/eta(n)/sigma^2(n))
E1 = POST_MEAN(AHr,AHc) - (1 - param.specCorr(BHr,BHc)) .* ...
    POST_MEAN(BHr,BHc);
E1 = (1./(param.specCorr(BHr,BHc) .* RHO(BHr,BHc))) .* E1;

kappa_upd = sum(E1, 1) + (POST_MEAN(1,g_ind) ./ ...
    param.specVar(1,g_ind));
kappa_upd = kappa_upd ./ ...
    ( sum(1./RHO(BHr,BHc), 1) + 1./param.specVar(1,g_ind) );

% ------------------------------------
%   Updated amplitude correlation, eta
% ------------------------------------

%mult_a = -2*N_g*(M-1);
mult_a = -2*(M-1);
mult_b = (RHO(2:M,g_ind)).^(-1) .* (2*real(GAMMA_CORR) - ...
    2*real(conj(param.specMean(2:M,g_ind)).*POST_MEAN(2:M,g_ind)) - ...
    2*(POST_VAR(1:M-1,g_ind) + abs(POST_MEAN(1:M-1,g_ind)).^2) + ...
    2*real(conj(param.specMean(1:M-1,g_ind)).*POST_MEAN(1:M-1,g_ind)));
mult_b = sum(mult_b);
mult_c = (RHO(2:M,g_ind)).^(-1) .* ((POST_VAR(2:M,g_ind) + ...
    abs(POST_MEAN(2:M,g_ind)).^2) - 2*real(GAMMA_CORR) + ...
    (POST_VAR(1:M-1,g_ind) + abs(POST_MEAN(1:M-1,g_ind)).^2));
mult_c = 2 * sum(mult_c);
%mult_d = -2*N_g;
mult_d = -2;
mult_e = (2./RHO(1,g_ind)) .* (POST_VAR(1,g_ind) + ...
    abs(POST_MEAN(1,g_ind)).^2 + abs(param.specMean(1,g_ind)).^2 - ...
    2*real(conj(param.specMean(1,g_ind)).*POST_MEAN(1,g_ind)));
% mult_e = mult_e;


try
    for n = 1:N
        alpha_roots = roots([-mult_a/2, (mult_a - (mult_b(n)/2 + ...
            mult_e(n)/2) + mult_d/2), ((mult_b(n) + mult_e(n)) - mult_c(n)/2), ...
            mult_c(n)]);
        if isempty(alpha_roots(alpha_roots > 0 & alpha_roots < 1))
            % Return previous estimate
            param.specCorr(:,n) = param.specCorr(:,n);
        else
            % Clip allowable range for alpha
            eta_upd = alpha_roots(alpha_roots > 0 & ...
                alpha_roots < 1);
            eta_upd = max(min(eta_upd, 0.99), 0.001);
            param.specCorr(:,n) = repmat(eta_upd, M, 1);
        end
    end
catch
% Either NaN or inf arguments were passed to roots fxn,
% suggesting that the EM procedure is diverging.  We can try to
% salvage it by just returning the previous estimate of alpha,
% but no guarantees here...
warning(['NaN or Inf arguments encountered during alpha update' ...
    ',thus returning previous estimate'])
end

param.specMean = resize(kappa_upd, M, N);
param.specVar = resize(sig_upd, M, N);
    
end

return