%Path setup for P-BiG-AMP

%Setup cvx
if exist('cvx_setup.m')==2
    curDir = pwd;
    cvxDir = fileparts(which('cvx_setup.m'));
    cd(cvxDir)
    cvx_setup
    cd(curDir)
    addpath([cvxDir,'/sedumi/'])
else
    error('cvx_setup.m is not on the path')
end

%Setup SEDUMI


%Add GAMPMATLAB paths
if exist('gampEst.m')==2
    gampDir = fileparts(fileparts(which('gampEst.m')))
    addpath([gampDir,'/phase']);
    addpath([gampDir,'/EMGMAMP']);
    addpath([gampDir,'/BiGAMP']);
    addpath([gampDir,'/examples/BiGAMP']);
else
    error('gampEst.m is not on the path')
end

%Add Zhu STLS
%if exist('WSSTLS_sdm_affine.m')~=2
%    error('WSSTLS_sdm_affine.m is not on the path')
%end

%Add TFOCS
if exist('tfocs.m')~=2
    error('tfocs.m is not on the path')
end

%Add tensor toolbox
if exist('tt_dimscheck.m')==2
    ttDir = fileparts(which('tt_dimscheck.m'));
    addpath([ttDir,'/met/'])
else
    error('the tensor toolbox is not on the path')
end



