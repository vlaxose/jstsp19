%Add Paths for DL experiments

%NOTE: some of the comparison codes require MEX files to be built. Compiled
%mex code for 64-bit MATLAB running on MAC OS X are included. These files
%will need to be built for other platforms


%Get absolute path to the folder containing this file
basePath = [fileparts(mfilename('fullpath')) filesep];


%GAMPMATLAB paths
addpath([basePath '../../EMGMAMP']) %EM code
addpath([basePath '../../BiGAMP']) %BiG-AMP code
addpath([basePath '../../main']) %main GAMPMATLAB code

%K-SVD codes
addpath([basePath '../../BiGAMP/comparison_codes/ompbox10']);
addpath([basePath '../../BiGAMP/comparison_codes/ksvdbox13']);

%ER-SpUD
addpath([basePath '../../BiGAMP/comparison_codes/spud']);
addpath([basePath '../../BiGAMP/comparison_codes/spud/ALM-JW']);
addpath([basePath '../../BiGAMP/comparison_codes/spud/ALM-JW/helpers']);

%SPAMS
addpath([basePath '../../BiGAMP/comparison_codes/spams-matlab/src_release']);
addpath([basePath '../../BiGAMP/comparison_codes/spams-matlab/build']);
setenv('MKL_NUM_THREADS','1')
setenv('MKL_SERIAL','YES')
setenv('MKL_DYNAMIC','NO')

%sparselab
addpath([basePath '../../BiGAMP/comparison_codes/sparseLab']);



