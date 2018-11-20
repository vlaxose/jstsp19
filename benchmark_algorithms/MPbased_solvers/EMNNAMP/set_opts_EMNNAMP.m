%***********NOTE************
% The user does not need to call this function directly.  It is called
% insead by EMNNAMP.
%
% This function checks if the GAMP and EM options are set, and if not, 
% set them to defaults.
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 8/01/13
% Change summary: 
%   v 1.0 (JV)- First release
%
% Version 1.0

%%
function [optALG, optEM, optGAMP] = set_opts_EMNNAMP(optALG_user, optEM_user, optGAMP_user, N, T)


%Set default EM options
[optALG,optEM] = EMNNOpt();

if isfield(optALG_user,'linEqMat')
    optALG.linEqMat = optALG_user.linEqMat;
else
    optALG.linEqMat = [];
end

if isfield(optEM_user,'linEqMeas')
    optALG.linEqMeas = optALG_user.linEqMat;
else
    optALG.linEqMeas = [];
end

%Change any algorithm options specified by user
if ~isempty(optALG_user)
   
    %Change the main EM options if specified by user
    names = fieldnames(optALG_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optALG), names{i}));
            optALG.(names{i}) = optALG_user.(names{i});
	else
	    warning(['''',names{i},''' is an unrecognized algorithm option'])
        end       
    end
end

%Change any EM options specified by user
if ~isempty(optEM_user)

    %Check to see if user set any other optional EM parameters
    if isfield(optEM_user,'noise_var'), 
      optEM.noise_var = optEM_user.noise_var; 
    end;
    if isfield(optEM_user,'outLapRate'), 
      optEM.outLapRate = optEM_user.outLapRate; 
    end;
    if isfield(optEM_user,'inExpRate'), 
      optEM.inExpRate = optEM_user.inExpRate; 
    end;
    if isfield(optEM_user,'tau'), 
      optEM.tau = optEM_user.tau; 
    end;
    if isfield(optEM_user,'active_weights'), 
      optEM.active_weights = optEM_user.active_weights; 
    end;
    if isfield(optEM_user,'active_loc'), 
      optEM.active_loc = optEM_user.active_loc; 
    end;
    if isfield(optEM_user,'active_scales'), 
      optEM.active_scales = optEM_user.active_scales; 
    end;
   
    %Change the main EM options if specified by user
    names = fieldnames(optEM_user);
    for i = 1:length(names) 
        if any(strcmp(fieldnames(optEM), names{i}));
            optEM.(names{i}) = optEM_user.(names{i});
	else
	    warning(['''',names{i},''' is an unrecognized EM option'])
        end       
    end
else
    optEM.linEqMat = ones(1,N);
    optEM.linEqMeas = ones(1,T);
end


%Set default GAMP options
optGAMP = GampOpt('nit',8,...
                    'removeMean',false,...
                    'adaptStep',true,...
                  'stepWindow',0,...
                    'bbStep',0,...
                    'uniformVariance',0,...
                    'verbose',0,...
                  'tol',1e-5,...
                    'step',1,...
                    'stepMin',0,...
                    'stepMax',1,...
                    'stepIncr',1.1,...
                    'stepDecr',0.5,...
                    'maxBadSteps',inf,...
                    'maxStepDecr',0.8,...
                    'xvarMin',1e-12,...
                    'pvarMin',1e-12,...
                  'stepTol',1e-4,...
                  'pvarStep',true,...
                    'varNorm',false,...
                    'valIn0',-Inf);
                
%Change any GAMP options specified by user
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


return