%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMNNAMP.
%
% This function returns real EM updates of the parameters tau, 
% omega, theta, and phi given the GAMP outputs Rhat and Rvar
%
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 8/1/12
% Change summary: 
%   v 1.0 (JV)- First release
%   v 2.0 (JV)- Accounts for MMV model.
%
% Version 2.0

%%
function [param] = NNGMupdate(Rhat, Rvar, param, optSIMP)

%Calculate Problem dimensions
L = size(param.active_weights,3);
[N, T] = size(Rhat);

beta = zeros(N,T,L); gamma = zeros(N,T,L);
nu = zeros(N,T,L); act = zeros(N,T,L);
tau = param.tau;


%Run through mixture components
for i = 1:L
    dummy = param.active_scales(:,:,i) + Rvar + eps;
    beta(:,:,i) = tau.*param.active_weights(:,:,i)...
        .*exp(-abs(Rhat-param.active_loc(:,:,i)).^2./2./dummy)./sqrt(dummy);
    gamma(:,:,i) = (Rvar.*param.active_loc(:,:,i)+param.active_scales(:,:,i).*Rhat)./dummy;
    nu(:,:,i) = param.active_scales(:,:,i).*Rvar./dummy;
    act(:,:,i) = sqrt(Rvar./(dummy)).*param.active_weights(:,:,i)...
    .*exp(((Rhat-param.active_loc(:,:,i)).^2./...
    abs(dummy)-abs(Rhat).^2./Rvar)./(-2));
end;  


%compute terms inside of cdf/pdf components
alpha = -gamma./sqrt(nu);

cdf_comp = erfc(alpha/sqrt(2))./erfc(-param.active_loc./sqrt(param.active_scales)/sqrt(2));
inv_mill = sqrt(2/pi)./erfcx(alpha/sqrt(2));

%find normalizing factor
zeta = repmat(sum(beta.*cdf_comp,3) + (1-tau)./...
     sqrt(Rvar).*exp(-abs(Rhat).^2./(2*Rvar)),[1 1 L]);
zeta(zeta ==0) = eps;
%zeta = repmat(sum(beta.*cdf_comp,3),[1 1 L]);
dummy = beta.*cdf_comp./zeta;
mn = (gamma + sqrt(nu).*inv_mill);

%Calculate updates to parameters
a_nt = param.tau./(1-param.tau+eps).*sum(act.*cdf_comp,3);
a_nt = 1./(1+a_nt.^(-1));
a_nt(isnan(a_nt)) = 0.001;

if optSIMP.learn_tau
    if strcmp(optSIMP.sig_dim,'joint')
        param.tau = sum(sum(a_nt))/N/T;
        param.tau = repmat(param.tau,[N T]);
    elseif strcmp(optSIMP.sig_dim,'row')
        param.tau = sum(a_nt,2)/T;
        param.tau = repmat(param.tau,[1 T]);
    elseif strcmp(optSIMP.sig_dim,'col')
        param.tau = sum(a_nt)/N;
        param.tau = repmat(param.tau,[N 1]);
    end
end

if strcmp(optSIMP.sig_dim,'joint')
    dummy2 = sum(sum(dummy,1),2);
    if optSIMP.learn_loc
        param.active_loc = sum(sum(dummy.*mn,1),2)./dummy2;
        param.active_loc = resize(param.active_loc,N,T,L);
    end
    if optSIMP.learn_scales
        param.active_scales = sum(sum(dummy.*(nu.*(1-inv_mill.*(inv_mill - alpha))+...
            (mn - param.active_loc).^2),1),2)./dummy2;
        param.active_scales = resize(param.active_scales,N,T,L);
    end
    if optSIMP.learn_weights
        param.active_weights = dummy2./ repmat(sum(dummy2,3),[1 1 L]);
        param.active_weights = resize(param.active_weights,N,T,L);
    end
elseif strcmp(optSIMP.sig_dim,'row')
    dummy2 = sum(dummy,2);
    if optSIMP.learn_loc
        param.active_loc = sum(dummy.*mn,2)./dummy2;
        param.active_loc = resize(param.active_loc,N,T,L);
    end
    if optSIMP.learn_scales
        param.active_scales = sum(dummy.*(nu.*(1-inv_mill.*(inv_mill - alpha))+...
            (mn - param.active_loc).^2),2)./dummy2;
        param.active_scales = resize(param.active_scales,N,T,L);
    end
    if optSIMP.learn_weights
        param.active_weights = dummy2./ repmat(sum(dummy2,3),[1 1 L]);
        param.active_weights = resize(param.active_weights,N,T,L);
    end
elseif strcmp(optSIMP.sig_dim,'col')
    dummy2 = sum(dummy,1);
    if optSIMP.learn_loc
        param.active_loc = sum(dummy.*mn,1)./dummy2;
        param.active_loc = resize(param.active_loc,N,T,L);
    end
    if optSIMP.learn_scales
        param.active_scales = sum(dummy.*(nu.*(1-inv_mill.*(inv_mill - alpha))+...
            (mn - param.active_loc).^2),1)./dummy2;
        param.active_scales = resize(param.active_scales,N,T,L);
    end
    if optSIMP.learn_weights
        param.active_weights = dummy2./ repmat(sum(dummy2,3),[1 1 L]);
        param.active_weights = resize(param.active_weights,N,T,L);
    end
end

return;