function options = MCOpt

% select mode
options.SPA = true;  % default mode is sum-product. set to false to run min-sum

% display options
options.plot_hist = 0; % plot gamp history? (0=no, #=figure number)
options.plot_wgt = 0; % plot weight vectors? (0=no, #=figure number)
options.verbose = true; % prints various things to the screen as mcGAMP runs

% preprocessing options
options.meanRemove = false; 
options.colNormalize = false; 

% parameter tuning options
options.tuneVar = true; % dflt=false
options.tuneMean = false; % dflt=false
options.tuneSpar = true; % dflt=true
options.tuneDelay = 0; % dflt=0

% error computation options
options.knowEmpPerrFxn = false; % empirical probability of error
options.knowThePerrFxn = false; % theoretical probability of error

% sum-prod initialization options
options.initTypeSP = 1; % in {0 = zeros, 1 = *scaled k-term approx} *scaled to match mnl likelihood
options.priorMeanTypeSP = 1; % in {0 = zero mean, 1 = xhat0}

% min-sum initialization options
options.priorLambdaMS = 1; % {0 = set lambda equal to corresponding laplacian lambda based on data, 
% 1 = set lambda so pvar = 1 at the first iteration (helps numerically),
% dflt = 1

% data model specification
options.x_bayes = [];
options.mu = [];
options.v = [];
options.Pbayes = nan;

% empirical test data (optional)
options.A_test = [];
options.y_test = [];

