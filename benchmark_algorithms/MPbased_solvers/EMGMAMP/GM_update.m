%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMGMAMP.
%
% This function returns real EM updates of the parameters lambda, 
% omega, theta, and phi given the GAMP outputs Rhat and Rvar
%
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 4/4/12
% Change summary: 
%   v 1.0 (JV)- First release
%   v 2.0 (JV)- Accounts for MMV model.
%
% Version 2.0
%%
function [lambda, omega, theta, phi, EMstate] = GM_update(Rhat, Rvar, lambda, omega, theta, phi, EMopt)

if any(~isreal(Rhat(:)))
    warning('Complex signal is being used with GM_update.  Please use CGM_update')
end

%Calcualte Problem dimensions
L = size(omega,3);
[N, T] = size(Rhat);

D_l = zeros(N,T,L); a_nl = zeros(N,T,L);
gamma = zeros(N,T,L); nu = zeros(N,T,L);

%Evaluate posterior likelihoods
for i = 1:L
    post_var_scale = Rvar+phi(:,:,i)+eps;
    D_l(:,:,i) = lambda.*omega(:,:,i)./sqrt(post_var_scale)...
        .*exp(-abs(theta(:,:,i)-Rhat).^2./(2*post_var_scale));
    gamma(:,:,i) = (Rhat.*phi(:,:,i)+Rvar.*theta(:,:,i))./post_var_scale;
    nu (:,:,i) = (Rvar.*phi(:,:,i))./post_var_scale;
    a_nl(:,:,i) = sqrt(Rvar./(post_var_scale)).*omega(:,:,i)...
    .*exp((abs(Rhat-theta(:,:,i)).^2./abs(post_var_scale)-abs(Rhat).^2./Rvar)./(-2));

end;

%Find posterior that the component x(n,t) is active
a_n = lambda./(1-lambda).*sum(a_nl,3);
a_n = 1./(1+a_n.^(-1));
a_n(isnan(a_n)) = 0.001;

if EMopt.learn_lambda
    if strcmp(EMopt.sig_dim,'joint')
        lambda = repmat(sum(sum(a_n))/N/T,[N,T]);
    elseif strcmp(EMopt.sig_dim,'col')
        lambda = repmat(sum(a_n)/N,[N 1]);
    elseif strcmp(EMopt.sig_dim,'row')
        lambda = repmat(sum(a_n,2)/T,[1 T]);
    end
end

%Find the Likelihood that component n,t belongs to class l and is active
E_l = D_l./repmat((sum(D_l,3)+(1-lambda)./sqrt(Rvar).*exp(-abs(Rhat).^2./(2*Rvar))),[1 1 L]);
%Ensure real valued probability
E_l(isnan(E_l)) = 0.999;

%Update parameters based on EM equations
if strcmp(EMopt.sig_dim,'joint')
    if ~(N == 1)
        N_l = sum(sum(E_l));
        if EMopt.learn_mean
            theta = resize(sum(sum(E_l.*gamma))./N_l,N,T,L);
        end
        if EMopt.learn_var
            phi = resize(sum(sum(E_l.*(nu+abs(gamma-theta).^2)))./N_l,N,T,L);
        end
        if EMopt.learn_weights
            omega = N_l/N/T;
            omega = omega./repmat(sum(omega, 3), [1, 1, L]);
        end
    end
elseif strcmp(EMopt.sig_dim,'col')
    if ~(N == 1)
        N_l = sum(E_l);
        if EMopt.learn_mean
            theta = resize(sum(E_l.*gamma)./N_l,N,T,L);
        end
        if EMopt.learn_var
            phi = resize(sum(E_l.*(nu+abs(gamma-theta).^2))./N_l,N,T,L);
        end
        if EMopt.learn_weights
            omega = N_l/N;
            omega = omega./repmat(sum(omega, 3), [1, 1, L]);
        end
    end
elseif strcmp(EMopt.sig_dim,'row')
    if ~(N == 1)
        N_l = sum(E_l,2);
        if EMopt.learn_mean
            theta = resize(sum(E_l.*gamma,2)./N_l,N,T,L);
        end
        if EMopt.learn_var
            phi = resize(sum(E_l.*(nu+abs(gamma-theta).^2),2)./N_l,N,T,L);
        end
        if EMopt.learn_weights
            omega = N_l/N;
            omega = omega./repmat(sum(omega, 3), [1, 1, L]);
        end
    end
end

omega = resize(omega,N,T,L);

EMstate.pi_n = a_n;
EMstate.beta_n = D_l;
EMstate.gamma_n = gamma;
EMstate.nu_n = nu;

return;
