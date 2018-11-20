function script_affine(whichK,whichM)

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


%Some options
%Control algorithms
optIn.tryPbigamp = 1;
optIn.tryEMPbigamp = 1;

%Specify real or complex-valued simulation
optIn.cmplx = true;

%Specify whether affine offset is added
optIn.affine = true;

%Specify a problem size
optIn.N = 100; %size of b and c

%Specify SNR
optIn.SNR = inf;

%Control uniformVariance
optIn.uniformVariance = 1;

%Define vectors of K and M
Kvals = round(linspace(10,100,10));
Mvals = round(linspace(10,300,10));

optIn.K = Kvals(whichK); %number of non-zeros in b and c
optIn.M = Mvals(whichM); %number of measurements


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
    results_point{trial} = trial_affine(optIn);
    
    
end


%Save the result
save(['./resultsAffine/affine_M_' num2str(whichM)...
    '_K_' num2str(whichK)])




