%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMGMAMP.
%
% This function checks if the GAMP and EM options are set, and if not, set
% them to defaults.
%
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 11/23/15
% Change summary: 
%   v 2.0 (JV)- First release
%   v 2.1 (JV)- Cleaner version
%   v 2.2 (JV)- increased max iters. Changed default maxBethe = true.
%   v 2.3 (PS)- Changed several defaults due to internal noise-variance learning
%
% Version 2.3
%
function [optGAMP, optEM] = check_opts(optGAMP_user, optEM_user)

%Set default GAMP options
optGAMP = GampOpt('nit',500,...
                    'removeMean',false,...
                    'removeMeanExplicit',true,...
                    'adaptStep',true,...
                    'adaptStepBethe',true,...
                  'stepWindow',5,...
                    'bbStep',0,...
                  'uniformVariance',true,...
                    'verbose',0,...
                    'tol',1e-5,...
                  'stepMax',1.0,...
                  'stepMin',0.2,...
                  'step',1.0,...
                    'stepIncr',1.1,...
                    'stepDecr',0.5,...
                    'pvarMin',1e-10,...
                    'xvarMin',0,...
                    'maxBadSteps',inf,...
                    'maxStepDecr',0.8,...
                    'stepTol',1e-10,...
                  'pvarStep',true,...
                    'rvarStep',false,...
                    'varNorm',false, ...
                    'valIn0',-Inf);
%Modify some defaults when in "robust_gamp" mode
if ~isempty(optEM_user),
    if isfield(optEM_user,'robust_gamp'),
        if optEM_user.robust_gamp,
            optGAMP.step = 0.1;
            optGAMP.stepMin = 0.1;
            optGAMP.stepMax = 0.5;
            optGAMP.nit = 1000;
            optGAMP.removeMean = true;
	end
    end
end
%Override GAMP defaults if specified by user
if ~isempty(optGAMP_user)
    names = fieldnames(optGAMP_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optGAMP), names{i}));
            optGAMP.(names{i}) = optGAMP_user.(names{i});
	else
	    warning(['''',names{i},''' is an unrecognized GAMP option'])
        end       
    end
end
%Force these GAMP options regardless of what user or default says!
optGAMP.legacyOut = false;
optGAMP.warnOut = false;


%Set default EM options
optEM = EMOpt();

%Change any EM options specified by user
if ~isempty(optEM_user)
    %Set GM order in accordance with user's heavy_tailed preference
    if isfield(optEM_user,'heavy_tailed')
        optEM.heavy_tailed = optEM_user.heavy_tailed;
    end;
    if ~isfield(optEM_user,'L') % ...if user has no preference about L
        if optEM.heavy_tailed
            optEM.L = 4;
        else
            optEM.L = 3;
        end
    end;

    %Check to see if user set any other optional EM parameters
    if isfield(optEM_user,'noise_var'), 
      optEM.noise_var = optEM_user.noise_var; 
    end;
    if isfield(optEM_user,'lambda'), 
      optEM.lambda = optEM_user.lambda; 
    end;
    if isfield(optEM_user,'active_weights'), 
      optEM.active_weights = optEM_user.active_weights; 
    end;
    if isfield(optEM_user,'active_mean'), 
      optEM.active_mean = optEM_user.active_mean; 
    end;
    if isfield(optEM_user,'active_var'), 
      optEM.active_var = optEM_user.active_var; 
    end;
    if isfield(optEM_user,'cmplx_in')
        optEM.cmplx_in = optEM_user.cmplx_in;
    end
    if isfield(optEM_user,'cmplx_out')
        optEM.cmplx_out = optEM_user.cmplx_out;
    end
%   if isfield(optEM_user,'heavy_tailed') && ~isfield(optEM_user,'hiddenZ')
%       optEM_user.hiddenZ = optEM_user.heavy_tailed;
%   end
   
    %Change the main EM options if specified by user
    names = fieldnames(optEM_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optEM), names{i}));
            optEM.(names{i}) = optEM_user.(names{i});
	else
	    warning(['''',names{i},''' is an unrecognized EM option'])
        end       
    end
end

return
