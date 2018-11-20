%Add Paths for Matrix Completion experiments

%NOTE: some of the comparison codes require MEX files to be built. Compiled
%mex code for 64-bit MATLAB running on MAC OS X are included. These files
%will need to be built for other platforms

%Get absolute path to the folder containing this file
basePath = [fileparts(mfilename('fullpath')) filesep];

%GAMPMATLAB paths
addpath([basePath '../../EMGMAMP']) %EM code
addpath([basePath '../../BiGAMP']) %BiG-AMP code
addpath([basePath '../../main']) %main GAMPMATLAB code

%Comparison codes
addpath([basePath '../../BiGAMP/comparison_codes/grouse']);
addpath([basePath '../../BiGAMP/comparison_codes/inexact_alm_mc']);
addpath([basePath '../../BiGAMP/comparison_codes/PROPACK_SVT']);
addpath([basePath '../../BiGAMP/comparison_codes/LMaFit_adp']);
addpath([basePath '../../BiGAMP/comparison_codes/LMaFit_adp/Utilities']);
addpath([basePath '../../BiGAMP/comparison_codes/VBLRMat']);
addpath([basePath '../../BiGAMP/comparison_codes/MatrixALPS_v0p1/ALPS']);
