%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by HUTAMP.
%
% This function returns real EM updates of the parameters lambda, 
% omega, theta, and phi given the GAMP outputs Rhat and Rvar
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

%%
function [stateFin] = abun_update(Rhat, Rvar, stateFin, optALG)

%Calculate Problem dimensions
[N, T] = size(Rhat);

%Find number of components and perallocate memory
L = size(stateFin.active_weights,3);
beta = zeros(N,T,L); gamma = zeros(N,T,L);
nu = zeros(N,T,L); act = zeros(N,T,L);
lambda = stateFin.lambda;


%Run through mixture components and compute needed quantities
for i = 1:L
    var_post = stateFin.active_scales(:,:,i) + Rvar + eps;
    beta(:,:,i) = lambda.*stateFin.active_weights(:,:,i)...
        .*exp(-abs(Rhat-stateFin.active_loc(:,:,i)).^2./2./var_post)./sqrt(var_post);
    gamma(:,:,i) = (Rvar.*stateFin.active_loc(:,:,i)+stateFin.active_scales(:,:,i).*Rhat)./var_post;
    nu(:,:,i) = stateFin.active_scales(:,:,i).*Rvar./var_post;
    act(:,:,i) = sqrt(Rvar./(var_post)).*stateFin.active_weights(:,:,i)...
    .*exp(((Rhat-stateFin.active_loc(:,:,i)).^2./...
    abs(var_post)-abs(Rhat).^2./Rvar)./(-2));
end;  

%compute arguments inside of cdf/pdf components
alpha = -gamma./sqrt(nu);

cdf_comp = erfc(alpha/sqrt(2))./erfc(-stateFin.active_loc./sqrt(stateFin.active_scales)/sqrt(2));
inv_mill = sqrt(2/pi)./erfcx(alpha/sqrt(2));

%find normalizing factor
zeta = repmat(sum(beta.*cdf_comp,3) + (1-lambda)./...
     sqrt(Rvar).*exp(-abs(Rhat).^2./(2*Rvar)),[1 1 L]);
zeta(zeta == 0) = eps;
%Compute posterior class activity probabilities
class_act = beta.*cdf_comp./zeta;
class_act(class_act == 0) = eps;
mn = (gamma + sqrt(nu).*inv_mill);

%Calculate updates to parameters
%Learn sparsity rate
if optALG.learn_lambda
    a_nt = stateFin.lambda./(1-stateFin.lambda+eps).*sum(act.*cdf_comp,3);
    a_nt = 1./(1+a_nt.^(-1));
    a_nt(isnan(a_nt)) = 0.001;
    stateFin.lambda = sum(a_nt,2)/T;
    stateFin.lambda = repmat(stateFin.lambda,[1 T]);
end

%Learn locations, scales, and weights
%Compute scaling constant
class_act_norm = sum(class_act,2);
if optALG.learn_loc
    stateFin.active_loc = sum(class_act.*mn,2)./class_act_norm;
    stateFin.active_loc = resize(stateFin.active_loc,N,T,L);
end
if optALG.learn_scales
    stateFin.active_scales = sum(class_act.*(nu.*(1-inv_mill.*(inv_mill - alpha))+...
        (mn - stateFin.active_loc).^2),2)./class_act_norm;
    stateFin.active_scales = max(1e-14, stateFin.active_scales);
    stateFin.active_scales = resize(stateFin.active_scales,N,T,L);
end
if optALG.learn_weights
    stateFin.active_weights = class_act_norm./ repmat(sum(class_act_norm,3),[1 1 L]);
    stateFin.active_weights = resize(stateFin.active_weights,N,T,L);
end

return;