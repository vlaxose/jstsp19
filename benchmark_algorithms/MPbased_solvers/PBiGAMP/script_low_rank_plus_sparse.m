
function script_low_rank_plus_sparse(whichR,whichLam)

%Add required paths
if ~isdeployed
    path_setup;
end

%Display svn info for logging purposes
system('svn info')

%Shuffle the random seed
rng('shuffle')

%Specifu number of trials
L_avg = 10;

%Decide on measurements
optIn.M = 10000;


%Control algorithms
optIn.tryPbigamp = 1;
optIn.tryEMPbigamp = 1;
optIn.tryTfocs = 1;

%Specify a problem size
optIn.Mq = 100;
optIn.Nq = 100;

%Control uniformVariance
optIn.uniformVariance = 1;


%Specify sparsity of Phi- number of non-zeroes per matrix
%Set to negative for dense Phi
optIn.phiSparsity = 50;

%Levels
optIn.nuw = [0 20^2/12]; %First entry is AWGN, second entry is variance of large outliers



%Define vectors of K and M
Rvals = round(linspace(3,30,10));
Lamvals = linspace(0.05,0.5,10);

%Set them
optIn.R = Rvals(whichR);
optIn.lambda = Lamvals(whichLam);

%Preallocate the trials
results_point = cell(L_avg,1);


%% Computation


%Record start time
tstart = tic;


%Iterations
for trial = 1:L_avg
    
    %Update on progress
    disp(['On trial ' num2str(trial) ' of ' num2str(L_avg)...
        '. Time elapsed: ' num2str(toc(tstart)/60)]);
    
    
    %Do the trial
    results_point{trial} = trial_LowRank_Plus_Sparse_Matrix_Recovery(optIn);
    
    
end


%Save the result
save(['./resultsLrps/lrps_R_' num2str(whichR)...
    '_Lam_' num2str(whichLam)])




