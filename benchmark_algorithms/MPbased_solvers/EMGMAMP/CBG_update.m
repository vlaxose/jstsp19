%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMBGAMP.
%
% This function returns complex EM updates of the parameters lambda, 
% theta, and phi given the GAMP outputs Rhat and Rvar
%
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 1/28/13
% Change summary: 
%   v 1.0 (JV)- First release
%   v 2.0 (JV)- Accounts for MMV model.  
%
% Version 2.1
%%
function [lambda, theta, phi, EMstate] = CBG_update(Rhat, Rvar, lambda, theta, phi, EMopt)

%Calcualte Problem dimensions
[N, T] = size(Rhat);

%Calculate posterior means and variances
post_var_scale = phi + Rvar + eps;
gamma_n = (Rhat.*phi+Rvar.*theta)./post_var_scale;
nu_n = Rvar.*phi./post_var_scale;

%Find posterior that the component x(n,t) is active
A_n = (1-lambda)./(lambda).*phi./nu_n.*exp((abs(Rhat-theta).^2./post_var_scale-abs(Rhat).^2./Rvar));
A_n = 1./(1+A_n);
A_n(isnan(A_n)) = 0.999;

%Update BG parameters
if EMopt.learn_mean
    if strcmp(EMopt.sig_dim,'joint')
        theta = sum(sum(A_n.*gamma_n))/sum(sum(A_n));
        theta = repmat(theta,[N T]);
    elseif strcmp(EMopt.sig_dim,'col')
        theta = sum(A_n.*gamma_n)./sum(A_n);
        theta = repmat(theta,[N 1]);
    elseif strcmp(EMopt.sig_dim,'row')
        theta = sum(A_n.*gamma_n,2)./sum(A_n,2);
        theta = repmat(theta,[1 T]);
    end
end

if EMopt.learn_var
    if strcmp(EMopt.sig_dim,'joint')
        phi = sum(sum(A_n.*(nu_n+abs(gamma_n-theta).^2)))/sum(sum(A_n));
        phi = repmat(phi,[N T]);
    elseif strcmp(EMopt.sig_dim,'col')
        phi = sum(A_n.*(nu_n+abs(gamma_n-theta).^2))./sum(A_n);
        phi = repmat(phi,[N 1]);
    elseif strcmp(EMopt.sig_dim,'row')
        phi = sum(A_n.*(nu_n+abs(gamma_n-theta).^2),2)./sum(A_n,2);
        phi = repmat(phi,[1 T]);
    end
end

if EMopt.learn_lambda
    if strcmp(EMopt.sig_dim,'joint')
        lambda= sum(sum(A_n))/N/T;
        lambda = repmat(lambda, [N T]);
    elseif strcmp(EMopt.sig_dim,'col')
        lambda = sum(A_n)/N;
        lambda = repmat(lambda, [N 1]);
    elseif strcmp(EMopt.sig_dim,'row')
        lambda = sum(A_n,2)/T;
        lambda = repmat(lambda, [1 T]);
    end
end

EMstate.pi_n = A_n;
EMstate.gamma_n = gamma_n;
EMstate.nu_n = nu_n;

return;