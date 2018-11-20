% --------------Note-------------------
% This function is to be called internally by HUTAMP.  This function sets
% the BiGAMP and EM options to their defaults, unless otherwise specified
% by the user.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 2/25/15
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

function [optALG, optBiGAMP] = set_opts_HUTAMP(optALG_user, optBiGAMP_user)

%Set default EM options
optALG = HUTOpt2();

%Change any EM options specified by user
if ~isempty(optALG_user)

    %Check to see if user set any other optional EM parameters
    if isfield(optALG_user,'noise_var'), 
      optALG.noise_var = optALG_user.noise_var; 
    end;
    if isfield(optALG_user,'lambda'), 
      optALG.lambda = optALG_user.lambda; 
    end;
    if isfield(optALG_user,'active_weights'), 
      optALG.active_weights = optALG_user.active_weights; 
    end;
    if isfield(optALG_user,'active_loc'), 
      optALG.active_loc = optALG_user.active_loc; 
    end;
    if isfield(optALG_user,'active_scales'), 
      optALG.active_scales = optALG_user.active_scales; 
    end;
    if isfield(optALG_user,'specCorr'), 
      optALG.specCorr = optALG_user.specCorr; 
    end;
    if isfield(optALG_user,'specMean'), 
      optALG.specMean = optALG_user.specMean; 
    end;
    if isfield(optALG_user,'specVar'), 
      optALG.specVar = optALG_user.specVar; 
    end;
    if isfield(optALG_user,'MRF_alpha'), 
      optALG.MRF_alpha = optALG_user.MRF_alpha; 
    end;
    if isfield(optALG_user,'MRF_betaH'), 
      optALG.MRF_betaH = optALG_user.MRF_betaH; 
    end;
    if isfield(optALG_user,'MRF_betaV'), 
      optALG.MRF_betaV = optALG_user.MRF_betaV; 
    end;
   
    %Change the main EM options if specified by user
    names = fieldnames(optALG_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optALG), names{i}));
            optALG.(names{i}) = optALG_user.(names{i});
	else
	    warning(['''',names{i},''' is an unrecognized EM option'])
        end       
    end
end

%If spatial coherence is enabled, do not learn lambda.
if optALG.spatialCoherence
    optALG.learn_lambda = false;
end

%Set default GAMP options
optBiGAMP = BiGAMPOpt('nit',20,...
                    'adaptStep',true,...
                    'stepWindow',inf,...
                    'uniformVariance',0,...
                    'verbose',0,...
                    'tol',1e-3,...
                    'step',0.01,...
                    'stepMin',0.01,...
                    'stepMax',0.5,...
                    'stepIncr',1.15,...
                    'stepDecr',0.5,...
                    'maxBadSteps',inf,...
                    'maxStepDecr',0.8,...
                    'AvarMin',1e-14,...
                    'xvarMin',1e-14,...
                    'pvarMin',1e-14,...
                    'pvarStep',true,...
                    'nitMin',4,...
                    'varNorm',false,...
                    'zvarToPvarMax', 1-1e-6);
                
%Change any GAMP options specified by user
if ~isempty(optBiGAMP_user)
    names = fieldnames(optBiGAMP_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optBiGAMP), names{i}));
            optBiGAMP.(names{i}) = optBiGAMP_user.(names{i});
	else
	    warning(['''',names{i},''' is an unrecognized BiGAMP option'])
        end       
    end
end

return