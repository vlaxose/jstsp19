%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMNNAMP.
%
% This function returns real EM updates of the signal's distributional 
% parameters
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 8/1/13
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

%%
function stateFin = EMNNAMP_in_update(Rhat, Rvar, stateFin, optALG, optEM)

%Calculate Problem dimensions
[N, T] = size(Rhat);

if strcmp(optALG.alg_type,'NNGMAMP')

    L = size(stateFin.active_weights,3);
    beta = zeros(N,T,L); gamma = zeros(N,T,L);
    nu = zeros(N,T,L); act = zeros(N,T,L);
    tau = stateFin.tau;


    %Run through mixture components
    for i = 1:L
        var_post = stateFin.active_scales(:,:,i) + Rvar + eps;
        beta(:,:,i) = tau.*stateFin.active_weights(:,:,i)...
            .*exp(-abs(Rhat-stateFin.active_loc(:,:,i)).^2./2./var_post)./sqrt(var_post);
        gamma(:,:,i) = (Rvar.*stateFin.active_loc(:,:,i)+stateFin.active_scales(:,:,i).*Rhat)./var_post;
        nu(:,:,i) = stateFin.active_scales(:,:,i).*Rvar./var_post;
        act(:,:,i) = sqrt(Rvar./(var_post)).*stateFin.active_weights(:,:,i)...
        .*exp(((Rhat-stateFin.active_loc(:,:,i)).^2./...
        abs(var_post)-abs(Rhat).^2./Rvar)./(-2));
    end;  

    %compute terms inside of cdf/pdf components
    alpha = -gamma./sqrt(nu);

    cdf_comp = erfc(alpha/sqrt(2))./erfc(-stateFin.active_loc./sqrt(stateFin.active_scales)/sqrt(2));
    inv_mill = sqrt(2/pi)./erfcx(alpha/sqrt(2));

    %find normalizing factor
    zeta = repmat(sum(beta.*cdf_comp,3) + (1-tau)./...
         sqrt(Rvar).*exp(-abs(Rhat).^2./(2*Rvar)),[1 1 L]);
    zeta(zeta == 0) = eps;
    %zeta = repmat(sum(beta.*cdf_comp,3),[1 1 L]);
    class_act = beta.*cdf_comp./zeta;
    class_act(class_act == 0) = eps;
    mn = (gamma + sqrt(nu).*inv_mill);

    %Calculate updates to stateFineters
    a_nt = stateFin.tau./(1-stateFin.tau+eps).*sum(act.*cdf_comp,3);
    a_nt = 1./(1+a_nt.^(-1));
    a_nt(isnan(a_nt)) = 0.001;

    if optEM.learn_tau
        if strcmp(optEM.inDim,'joint')
            stateFin.tau = sum(sum(a_nt))/N/T;
            stateFin.tau = repmat(stateFin.tau,[N T]);
        elseif strcmp(optEM.inDim,'row')
            stateFin.tau = sum(a_nt,2)/T;
            stateFin.tau = repmat(stateFin.tau,[1 T]);
        elseif strcmp(optEM.inDim,'col')
            stateFin.tau = sum(a_nt)/N;
            stateFin.tau = repmat(stateFin.tau,[N 1]);
        end
    end

    if strcmp(optEM.inDim,'joint')
        class_act_norm = sum(sum(class_act,1),2);
        if optEM.learn_loc
            stateFin.active_loc = sum(sum(class_act.*mn,1),2)./class_act_norm;
            stateFin.active_loc = resize(stateFin.active_loc,N,T,L);
        end
        if optEM.learn_scales
            stateFin.active_scales = sum(sum(class_act.*(nu.*(1-inv_mill.*(inv_mill - alpha))+...
                (mn - stateFin.active_loc).^2),1),2)./class_act_norm;
            stateFin.active_scales = resize(stateFin.active_scales,N,T,L);
        end
        if optEM.learn_weights
            stateFin.active_weights = class_act_norm./ repmat(sum(class_act_norm,3),[1 1 L]);
            stateFin.active_weights = resize(stateFin.active_weights,N,T,L);
        end
    elseif strcmp(optEM.inDim,'row')
        class_act_norm = sum(class_act,2);
        if optEM.learn_loc
            stateFin.active_loc = sum(class_act.*mn,2)./class_act_norm;
            stateFin.active_loc = resize(stateFin.active_loc,N,T,L);
        end
        if optEM.learn_scales
            stateFin.active_scales = sum(class_act.*(nu.*(1-inv_mill.*(inv_mill - alpha))+...
                (mn - stateFin.active_loc).^2),2)./class_act_norm;
            stateFin.active_scales = resize(stateFin.active_scales,N,T,L);
        end
        if optEM.learn_weights
            stateFin.active_weights = class_act_norm./ repmat(sum(class_act_norm,3),[1 1 L]);
            stateFin.active_weights = resize(stateFin.active_weights,N,T,L);
        end
    elseif strcmp(optEM.inDim,'col')
        class_act_norm = sum(class_act,1);
        if optEM.learn_loc
            stateFin.active_loc = sum(class_act.*mn,1)./class_act_norm;
            stateFin.active_loc = resize(stateFin.active_loc,N,T,L);
        end
        if optEM.learn_scales
            stateFin.active_scales = sum(class_act.*(nu.*(1-inv_mill.*(inv_mill - alpha))+...
                (mn - stateFin.active_loc).^2),1)./class_act_norm;
            stateFin.active_scales = resize(stateFin.active_scales,N,T,L);
        end
        if optEM.learn_weights
            stateFin.active_weights = class_act_norm./ repmat(sum(class_act_norm,3),[1 1 L]);
            stateFin.active_weights = resize(stateFin.active_weights,N,T,L);
        end
        stateFin.active_scales = max(stateFin.active_scales,optEM.minScale);
    end
%Update rate parameter if input distribution is laplacian
elseif  strcmp(optALG.alg_type,'NNLAMP')
    
    if optEM.learn_inExpRate;
        
        muU = Rhat - stateFin.inExpRate.*Rvar; %Find posterior mean of Gaussian
        alpha = -muU./sqrt(Rvar); %Find parameter for gaussian cdf term
        inv_mill = sqrt(2/pi)./erfcx(alpha/sqrt(2)); %Calculate inverse mills ratio
        expt = muU + sqrt(Rvar).*inv_mill; %Calculate final expected value of posterior
        
        %Find EM update of rate parameter
        if strcmp(optEM.inDim,'joint')
            stateFin.inExpRate = T*N./sum(sum(expt));
        elseif strcmp(optEM.inDim,'row')
            stateFin.inExpRate = T./sum(expt,2);
        elseif strcmp(optEM.inDim,'col')
            stateFin.inExpRate = N./sum(expt,1);
        end
        
        %Resize final estimate to size of signal.
        stateFin.inExpRate = resize(stateFin.inExpRate,N,T);
    end
end

return