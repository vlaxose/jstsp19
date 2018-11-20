
function script_calibration(whichK,whichNb)

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
optIn.tryWSSTLS = 0;
optIn.trySparseLift = 1;

%Specify a problem size
optIn.Nc = 256; %length of signal
optIn.M = 128; %number of measurements

%Specify SNR
optIn.SNR = inf;

%Control uniformVariance
optIn.uniformVariance = 1;

%WSS-STLS options
optIn.stls_lam_steps = 1;

%Define vectors of K and M
Kvals = round(linspace(2,60,10));
Nbvals = round(linspace(2,40,10));

optIn.K = Kvals(whichK); %number of non-zeros in b and c
optIn.Nb = Nbvals(whichNb); %number of measurements


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
    results_point{trial} = trial_calibration(optIn);
    
    
end


%Save the result
save(['./resultsCalibration/calibration_Nb_' num2str(whichNb)...
    '_K_' num2str(whichK)])




