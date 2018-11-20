% CLASS: GUIData
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The GUIData class contains a variety of properties that provide
%   information and methods needed for the DemoGUI classification GUI to
%   operate, and also contains the information on the many components that
%   collectively define a particular GAMP configuration (i.e.,
%   information on EstimIn, EstimOut, and LinTrans classes, as well as 
%   runtime options (the GampOpt class)) along with the particular datasets 
%   being used for testing purposes.
%
% PROPERTIES (State variables)
%   EstimInUIData           A structure containing the names of available
%                           EstimIn classes, parameters associated with
%                           those classes, default and current values of
%                           those parameters, and an indicator of the
%                           currently selected (in the GUI) EstimIn class
%   EstimOutUIData
%   GAMPOptUIData
%   DatasetUIData
%
% METHODS (Subroutines/functions)
%   GUIData()
%       - Default constructor that initializes all properties to their
%         default values.
%   [EstimIn, EstimOut, LinTrans, GampOpt] = GenGAMPObjs(obj)
%       - Generates the objects that GAMP requires to run
%   [A, y, x_true] = GenDataset(obj)
%       - Generates synthetic dataset, or returns user-provided one, based
%         on the state of a GUIData object, obj.
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 11/25/12
% Change summary: 
%       - Created (11/24/12; JAZ)
% Version 0.1
%

classdef GUIData < hgsetget

    properties
        % Declare the properties
        EstimInUIData = struct();
        EstimOutUIData = struct();
        GAMPOptUIData = struct();
        DatasetUIData = struct();
    end % properties
    
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = GUIData()
            % Initialize a default object
            
            % EstimIn UI data
            % *************************************************************
            obj.EstimInUIData.Names = {
                'Gaussian Prior'
                'Complex Gaussian Prior'
                'Elastic Net Prior'
                'Ell-p Norm Prior'
                'Ell-p Norm Prior (DMM)'
%                 'Gaussian Mixture Prior'
%                 'Complex Gaussian Mixture'
%                 'Multinormial Prior'
%                 'Non-coherent CAWGN Prior'
%                 'Non-negative Gaussian Mixture'
%                 'Null Prior (Noninformative)'
%                 'Soft-Threshold Prior (Ell-1)'
%                 'Soft-Threshold Prior (DMM)'
                };
            obj.EstimInUIData.ClassNames = {
                'AwgnEstimIn'
                'CAwgnEstimIn'
                'ElasticNetEstimIn'
                'EllpEstimIn'
                'EllpDMMEstimIn'
%                 'GMEstimIn'
%                 'CGMEstimIn'
%                 'DisScaEstim'
%                 'ncCAwgnEstimIn'
%                 'NNGMEstimIn'
%                 'NullEstimIn'
%                 'SoftThreshEstimIn'
%                 'SoftThreshDMMEstimIn'
                };
            obj.EstimInUIData.selectedEstimIn = 1;      % Current selection (index into Names)
            obj.EstimInUIData.Sparsify = false;         % Sparsify EstimIn class (i.e., wrap with SparseScaEstim)?
            obj.EstimInUIData.SparsityRate = 0.05;      % SparseScaEstim sparsity rate
            obj.EstimInUIData.Params = cell(13,1);      % 1 cell for each EstimIn class
            obj.EstimInUIData.Values = cell(13,1);      % 1 cell for each EstimIn class
            % Gaussian prior parameters and values
            obj.EstimInUIData.Params{1} = {
                % property name,        ui type         display text                    allowable
                'mean0',                'Edit',         'Prior mean (-Inf,Inf):  ',     'Mean of Gaussian distribution';
                'var0',                 'Edit',         'Prior variance (0,Inf):  ',    'Variance of Gaussian distribution';
                'maxSumVal',            'Checkbox',     'Max-sum (MAP) GAMP?  ',        'Check to run max-sum (MAP) version of GAMP'
                };
            obj.EstimInUIData.Values{1} = {
                % current value         default value
                0,                      0;
                1,                      1;
                false,                  0;
                };
            % Complex Gaussian prior parameters and values
            obj.EstimInUIData.Params{2} = {
                % property name,        ui type         display text                    allowable
                'mean0',                'Edit',         'Complex prior mean:  ',        'Complex mean of Gaussian distribution';
                'var0',                 'Edit',         '(Symmetric) prior variance (0,Inf):',  'Symmetric variance of Gaussian distribution';
                'maxSumVal',            'Checkbox',     'Max-sum (MAP) GAMP?  ',        'Check to run max-sum (MAP) version of GAMP'
                };
            obj.EstimInUIData.Values{2} = {
                % current value         default value
                0,                      0;
                1,                      1;
                false,                  0;
                };
            % Elastic Net prior parameters and values
            obj.EstimInUIData.Params{3} = {
                % property name,        ui type         display text                        allowable
                'lambda1',              'Edit',         'Ell-1 penalty (0,Inf):  ',         'Value of ell-1 penalty term';
                'lambda2',              'Edit',         'Ell-2 penalty (0,Inf):  ',         'Value of ell-2 penalty term';
                'maxSumVal',            'Checkbox',     'Max-sum (MAP) GAMP?  ',            'Check to run max-sum (MAP) version of GAMP';
                };
            obj.EstimInUIData.Values{3} = {
                % current value         default value
                sqrt(2),                sqrt(2);
                1/2,                    1/2;
                false,                  0;
                };
            % Ell-p norm prior parameters and values
            obj.EstimInUIData.Params{4} = {
                % property name,        ui type         display text                        allowable
                'lambda',               'Edit',         'Ell-p penalty gain (0,Inf):  ',    'Langrange multiplier/penalty term in MAP cost';
                'p',                    'Edit',         'Choice of p-norm (0,1]:  ',        'Choice of p-norm';
                'mean0',                'Edit',         'Initial mean (-Inf,Inf):  ',       'Recommended to leave at default';
                'var0',                 'Edit',         'Initial variance (0,Inf):  ',      'Recommended to leave at default';
                };
            obj.EstimInUIData.Values{4} = {
                % current value         default value
                1,                      1;
                1,                      1;
                0,                      0;
                5e-4,                   5e-4;
                };
            % Ell-p (DMM) norm prior parameters and values
            obj.EstimInUIData.Params{5} = {
                % property name,        ui type         display text                        allowable
                'alpha',                'Edit',         'DMM threshold gain (0,Inf):  ',    'Donoho/Maleki/Montanari threshold gain (alpha)';
                'p',                    'Edit',         'Choice of p-norm (0,1]:  ',        'Choice of p-norm';
                };
            obj.EstimInUIData.Values{5} = {
                % current value         default value
                1,                      1;
                1,                      1;
                0,                      0;
                5e-4,                   5e-4;
                };
            
            
            % EstimOut UI Data
            % *************************************************************
            obj.EstimOutUIData.Names = {
                'AWGN Channel'
                'Complex AWGN'
                'Gaussian-Mixture'
                'Complex G.M.'
                };
            obj.EstimOutUIData.ClassNames = {
                'AwgnEstimOut'
                'CAwgnEstimOut'
                'GaussMixEstimOut'
                'CGaussMixEstimOut'
                };
            obj.EstimOutUIData.selectedEstimOut = 1;	% Current selection (index into Names)
            obj.EstimOutUIData.Params = cell(4,1);       % 1 cell for each EstimOut class
            obj.EstimOutUIData.Values = cell(4,1);       % 1 cell for each EstimOut class
            % AWGN channel parameters and values
            obj.EstimOutUIData.Params{1} = {
                % property name,        ui type         display text                    allowable
                'y',                    'Meas',         '',                             '',
                'wvar',                 'Edit',         'Noise variance (0,Inf):  ',    'Enter the additive Gaussian noise variance';
                'scale',                'Edit',         'Channel gain (0,Inf):  ',      'Recommended to leave at default value';
                'maxSumVal',            'Checkbox',     'Max-sum (MAP) GAMP?  ',        'Check to run max-sum (MAP) version of GAMP';
                };
            obj.EstimOutUIData.Values{1} = {
                % current value         default value
                [],                     [];
                1e-2,                   1e-2;
                1,                      1;
                false,                  0;
                };
            % Complex AWGN channel parameters and values
            obj.EstimOutUIData.Params{2} = {
                % property name,        ui type         display text                    allowable
                'y',                    'Meas',         '',                             '',
                'wvar',                 'Edit',         'Noise variance (0,Inf):  ',    'Enter the (symmetric) Gaussian noise variance';
                'scale',                'Edit',         'Channel gain (0,Inf):  ',      'Recommended to leave at default value';
                'maxSumVal',            'Checkbox',     'Max-sum (MAP) GAMP?  ',        'Check to run max-sum (MAP) version of GAMP';
                };
            obj.EstimOutUIData.Values{2} = {
                % current value         default value
                [],                     [];
                1e-2,                   1e-2;
                1,                      1;
                false,                  0;
                };
            % Gaussian mixture channel parameters and values
            obj.EstimOutUIData.Params{3} = {
                % property name,        ui type         display text                    allowable
                'Y',                    'Meas',         '',                             '',
                'nu0',                  'Edit',         'Small noise variance (0,Inf):  ','Enter the smaller of the two variances';
                'nu1',                  'Edit',         'Large noise variance (0,Inf):  ','Enter the larger of the two variances';
                'lambda',               'Edit',         'Large-component prob. [0,1]:  ', 'Enter probability that noise is generated from larger variance component';
                };
            obj.EstimOutUIData.Values{3} = {
                % current value         default value
                [],                     [];
                1e-2,                   1e-2;
                1,                      1;
                0.05,                   0.05;
                };
            % Complex Gaussian mixture channel parameters and values
            obj.EstimOutUIData.Params{4} = {
                % property name,        ui type         display text                    allowable
                'Y',                    'Meas',         '',                             '',
                'nu0',                  'Edit',         'Small noise variance (0,Inf):  ','Enter the smaller of the two variances';
                'nu1',                  'Edit',         'Large noise variance (0,Inf):  ','Enter the larger of the two variances';
                'lambda',               'Edit',         'Large-component prob. [0,1]:  ', 'Enter probability that noise is generated from larger variance component';
                };
            obj.EstimOutUIData.Values{4} = {
                % current value         default value
                [],                     [];
                1e-2,                   1e-2;
                1,                      1;
                0.05,                   0.05;
                };
            
            
            
            % GAMPOptUIdata
            % *************************************************************
            % GAMP-related parameters
            obj.GAMPOptUIData.GAMPParams = {
                % property name,        ui type         display text                            allowable                                                                                                                       subpanel (1=Basic,2=Int.,3=Adv.)
                'nit',                  'Edit',         'Maximum # of GAMP iterations:  ',      'Enter the maximum number of GAMP iterations allowed',                                                                          1;
                'adaptStep',            'Checkbox',     'Enable adaptive step-sizing:  ',       'Check to enable adaptive step-sizing (useful for highly correlated measurement matrices',                                      1;
                'verbose',              'Checkbox',     'Run GAMP verbosely?  ',                'Check to have GAMP output diagnostic data to command window',                                                                  1;
                'tol',                  'Edit',         'Early termination tolerance:  ',       'Specify tolerance at which GAMP quits early',                                                                                  1;
                'uniformVariance',      'Checkbox',     'Use common (uniform) variance?  ',     'Check to restrict GAMP to using a single scalar message variance',                                                             1;
                'varNorm',              'Checkbox',     'Normalize variance messages?  ',       'Check to enable variance normalization for improved numerical precision',                                                      2;
                'stepMin',              'Edit',         'Minimum GAMP step size:  ',            'If adaptive step-sizing is enable, this sets the smallest allowable step',                                                     2;
                'stepMax',              'Edit',         'Maximum GAMP step size:  ',            'If adaptive step-sizing is enable, this sets the largest allowable step',                                                      1;
                'stepIncr',             'Edit',         'Multiplicative step increment:  ',     'If adaptive step-sizing is enable, this sets amount by which the step size increases if successful on last iteration',         2;
                'stepDecr',             'Edit',         'Multiplicative step decrement:  ',     'If adaptive step-sizing is enable, this sets amount by which the step size decreases if unsuccessful on last iteration',       2;
                'removeMean',           'Checkbox',     'Remove mean from A?  ',                'Check to remove mean from A by creating a new matrix with one additional row and column',                                      3;
                'pvarMin',              'Edit',         'Minimum output channel variance:  ',   'Specify (relative) minimum output channel variance, to prevent poor conditioning.',                                            3;
                'xvarMin',              'Edit',         'Minimum input channel variance:  ',    'Specify (relative) minimum input channel variance, to prevent poor conditioning.',                                             3;
                'stepWindow',           'Edit',         'Step-sizing window length:  ',         'Adjusts the length of the window used in step-sizing acceptance criteria.  Higher values lead to more stringency',             3;
                'step',                 'Edit',         'Initial (or static) step length:  ',   'Specify the initial step length (if adaptive step-sizing) or the static length (non-adaptive steps)',                          2;
                'stepTol',              'Edit',         'Early termination tolerance:  ',       'Iterations are terminated when the step size becomes smaller than this value. Set to -1 to disable',                           2;
                'pvarStep',             'Checkbox',     'Include step-size in pvar calcs?  ',   'Logical flag to include a step size in the pvar/zvar calculation. This momentum term often improves numerical performance',    2;
                'bbStep',               'Checkbox',     'Use Barzilai-Borwein step size?  ',    'Check to use a Barzilai-Borwein-type rule to choose a new step size after each successful step',                               3
                };
            obj.GAMPOptUIData.GAMPValues = {
                % current value         default value
                200,                  	200;
                false,                  false;
                false,                  false;
                1e-4,                   1e-4;
                false,                  false;
                true,                   true;
                0,                      0;
                1,                      1;
                1,                      1;
                0.5,                    0.5;
                false,                  false;
                1e-10,                  1e-10;
                1e-10,                  1e-10;
                20,                     20;
                1,                      1;
                1e-10,                  1e-10;
                true,                   true;
                false,                  false
                };
            
            
            % DatasetUIData setup
            obj.DatasetUIData.GenSynthData = true;  % Generate synthetic dataset flag
            obj.DatasetUIData.selectedAltEstimIn = 1;   % Index of alternative EstimIn class (if in use)
            % Synthetic dataset-related parameters
            obj.DatasetUIData.SynthParams = {
                % property name,        ui type         display text                            allowable
                'N',                    'Edit',         '# of unknowns (N):',                   'Dimensionality of the unknown signal, x';
                'M',                    'Edit',         '# of observations (M):',               'Dimensionality of the measurement vector, y';
                'Atype',                'PopupMenu',    'Measurement matrix type:',             {'Random Gaussian', 'Complex Gaussian', 'Uniform Random', 'Zero-One Random', 'Sparse Random', 'Row-subsampled DFT', 'Block-subsampled DFT'};
                'GenDiff',            	'Checkbox',     'Non-EstimIn signal generation?',       sprintf('%s\n %s', 'Check this box to generate the true signal from a distribution', 'other than the chosen EstimIn class (or with different parameters)')
                };
            obj.DatasetUIData.SynthValues = {
                % current value         default value
                1024,                  	1024;
                256,                   	256;
                'Random Gaussian',      1;
                false,                  false;
                };
            % Alternative EstimIn choice parameters
            obj.DatasetUIData.Sparsify = false;         % Sparsify EstimIn class (i.e., wrap with SparseScaEstim)?
            obj.DatasetUIData.SparsityRate = 0.05;      % SparseScaEstim sparsity rate
            obj.DatasetUIData.AltEstimIn = {};
            obj.DatasetUIData.WSpcValues = {
                % current value         default value       variable
                NaN,                    NaN;                % A
                NaN,                    NaN;                % y
                NaN,                    NaN;                % x_true
                };
        end
        
        
        % *****************************************************************
        %                     GENERATE DATASET METHOD
        % *****************************************************************
        % This method is called by DemoGUI.  It uses the current state of
        % the GUIData object, obj, to produce a LinTrans object, and
        % observations.  This data can either be generated synthetically, or can
        % simple involve a retrieval of variables already present in the
        % user's workspace.
        function [A, y, x_true, exitFlag] = generateDataset(obj)
            [A, y, x_true] = deal(NaN);
            exitFlag = 0;   % No error exit flag
            
            if obj.DatasetUIData.GenSynthData
                % Create an EstimIn object that will be used to generate
                % the synthetic true x
                if obj.DatasetUIData.SynthValues{4,1}
                    % Must generate true x from an EstimIn object that is
                    % different from the one used to recover x
                    [EstimInObj, exitFlag] = obj.createEstimInObj('alt');
                else
                    % Use same EstimIn object for both data generation and
                    % recovery
                    [EstimInObj, exitFlag] = obj.createEstimInObj('std');
                end
                if exitFlag ~= 0, return; end
                
                % Create an EstimOut object that will be used to generate
                % the synthetic y
                [EstimOutObj, exitFlag] = obj.createEstimOutObj();
                if exitFlag ~= 0, return; end

                % Get dimensions of synthetic dataset, and type of LinTrans
                % class to construct
                try
                    N = obj.DatasetUIData.SynthValues{strcmp('N', ...
                        obj.DatasetUIData.SynthParams(:,1)),1};
                    M = obj.DatasetUIData.SynthValues{strcmp('M', ...
                        obj.DatasetUIData.SynthParams(:,1)),1};
                    Atype = obj.DatasetUIData.SynthValues{strcmp('Atype', ...
                        obj.DatasetUIData.SynthParams(:,1)),1};
                    DataOptObj.N = N;
                    DataOptObj.M = M;
                    DataOptObj.Atype = Atype;
                catch ME
                    exitFlag = 1;   % Flag an error for calling fxn
                    msgStr = sprintf(['Failed to get Dataset options ' ...
                        'successfully.  Error message: %s'], ME.message);
                    msgbox(msgStr, 'Error with user-provided input', 'error');
                    return
                end

                % Generate synthetic training and test data
                [A, y, x_true, exitFlag] = obj.GenerateData(DataOptObj, ...
                    EstimInObj, EstimOutObj);
                if exitFlag ~= 0, return; end
                
            else
                % Get training and test data from user workspace
                A = obj.DatasetUIData.WSpcValues{1,1};
                y = obj.DatasetUIData.WSpcValues{2,1};
                x_true = obj.DatasetUIData.WSpcValues{3,1};
            end
        end
        
        
        % *****************************************************************
        %                        RUN GAMP METHOD
        % *****************************************************************
        % This method is called by DemoGUI.  It uses the current state of
        % the GUIData object, obj, to configure GAMP and execute it on the
        % dataset specified by an observation vector, y, and a LinTrans
        % object, A.  It returns the recovered signal estimate, x_gamp
        function [x_gamp, exitFlag] = RunGAMP(obj, y, A)
            x_gamp = NaN;
            
            % Create an EstimIn object (from EstimIn tab specs)
            [EstimInObj, exitFlag] = obj.createEstimInObj('std');
            if exitFlag ~= 0, return; end
            
            % Create an EstimOut object
            [EstimOutObj, exitFlag] = obj.createEstimOutObj();
            if exitFlag ~= 0, return; end
            
            % Place measurements into EstimOut object
            idx1 = obj.EstimOutUIData.selectedEstimOut;
            idx2 = strcmpi('Meas', obj.EstimOutUIData.Params{idx1}(:,2));
            var_name = obj.EstimOutUIData.Params{idx1}{idx2,1};
            try
                set(EstimOutObj, var_name, y);
            catch ME
                exitFlag = 1;   % Flag an error for calling fxn
                msgStr = sprintf(['Failed to store measurements in EstimOut object' ...
                    ' successfully.  Error message: %s'], ME.message);
                msgbox(msgStr, 'Error running GAMP', 'error');
                return
            end
            
            % Next, build a GampOpt object
            [GampOptObj, exitFlag] = obj.createGampOptObj();
            if exitFlag ~= 0, return; end
            
            % Now, run GAMP once
            try
                [x_gamp, xvarFinal, rhatFinal, rvarFinal,...
                    shatFinal, svarFinal, zhatFinal,zvarFinal,estHist] = ...
                    gampEst(EstimInObj, EstimOutObj, A, GampOptObj);
            catch ME
                exitFlag = 1;   % Flag an error for calling fxn
                msgStr = sprintf(['Failed to execute GAMP' ...
                    ' successfully.  Error message: %s'], ME.message);
                msgbox(msgStr, 'Error running GAMP', 'error');
                return
            end
        end
    end     % methods
    
    
    methods (Access = private)
        % *****************************************************************
        %                   CREATE ESTIMIN OBJ METHOD
        % *****************************************************************
        % This method is called by other public methods of the GUIData
        % class, and is used to create an EstimIn object based on the data
        % stored in the GUIData object, obj
        function [EstimInObj, exitFlag] = createEstimInObj(obj, tag)
            
            exitFlag = 0;               % No errors exit flag
            
            if strcmp(tag, 'std')
                % Generate an EstimIn object according to EstimIn tab specs 
                Idx = obj.EstimInUIData.selectedEstimIn;  % Index to chosen EstimIn class
                SelectedClassName = obj.EstimInUIData.ClassNames{Idx};   % Class name
                EstimInObj = feval(SelectedClassName);    % Construct default object
                % Populate the default object with the values contained in
                % the EstimInUIData property
                if ~isempty(obj.EstimInUIData.Params{Idx})
                    for i = 1:numel(obj.EstimInUIData.Params{Idx}(:,1))
                        value = obj.EstimInUIData.Values{Idx}{i,1};   % Current value
                        % Set parameter value
                        try
                            set(EstimInObj, obj.EstimInUIData.Params{Idx}{i,1}, value);
                        catch ME
                            exitFlag = 1;   % Flag an error for calling fxn
                            msgStr = sprintf(['Failed to build EstimIn object ' ...
                                '%s successfully.  Error message: %s'], ...
                                SelectedClassName, ME.message);
                            msgbox(msgStr, 'Error with user-provided input', 'error');
                            return
                        end
                    end
                end
                % Now we have the EstimIn object.  If the user wanted us to
                % sparsify this prior (i.e., make it a Bernoulli-* prior),
                % do that now
                if obj.EstimInUIData.Sparsify
                    EstimInObj = SparseScaEstim(EstimInObj, ...
                        obj.EstimInUIData.SparsityRate);
                end
            elseif strcmp(tag, 'alt')
                % Instead of obeying the EstimIn tab specs, we need to
                % construct an alternative EstimIn object based on specs
                % from the Dataset tab
                Idx = obj.DatasetUIData.selectedAltEstimIn;     % Index to chosen alt. EstimIn class
                SelectedClassName = obj.EstimInUIData.ClassNames{Idx};   % Class name
                EstimInObj = feval(SelectedClassName);          % Construct default object
                % Populate the default object with the values contained in
                % the EstimInUIData property
                if ~isempty(obj.DatasetUIData.AltEstimIn)
                    for i = 1:size(obj.DatasetUIData.AltEstimIn, 1)
                        value = obj.DatasetUIData.AltEstimIn{i,2};   % Current value
                        % Set parameter value
                        try
                            set(EstimInObj, obj.DatasetUIData.AltEstimIn{i,1}, value);
                        catch ME
                            exitFlag = 1;   % Flag an error for calling fxn
                            msgStr = sprintf(['Failed to build alternative EstimIn object ' ...
                                '%s successfully.  Error message: %s'], ...
                                SelectedClassName, ME.message);
                            msgbox(msgStr, 'Error with user-provided input', 'error');
                            return
                        end
                    end
                end
                % Now we have the EstimIn object.  If the user wanted us to
                % sparsify this prior (i.e., make it a Bernoulli-* prior),
                % do that now
                if obj.DatasetUIData.Sparsify
                    EstimInObj = SparseScaEstim(EstimInObj, ...
                        obj.DatasetUIData.SparsityRate);
                end
            end
        end
        
        
        % *****************************************************************
        %                   CREATE ESTIMOUT OBJ METHOD
        % *****************************************************************
        % This method is called by other public methods of the GUIData
        % class, and is used to create an EstimOut object based on the data
        % stored in the GUIData object, obj
        function [EstimOutObj, exitFlag] = createEstimOutObj(obj, tag)
            
            exitFlag = 0;               % No errors exit flag
            
            % Generate an EstimOut object according to EstimOut tab specs
            Idx = obj.EstimOutUIData.selectedEstimOut;  % Index to chosen EstimIn class
            SelectedClassName = obj.EstimOutUIData.ClassNames{Idx};   % Class name
            EstimOutObj = feval(SelectedClassName);    % Construct default object
            % Populate the default object with the values contained in
            % the EstimOutUIData property
            if ~isempty(obj.EstimOutUIData.Params{Idx})
                for i = 1:numel(obj.EstimOutUIData.Params{Idx}(:,1))
                    value = obj.EstimOutUIData.Values{Idx}{i,1};   % Current value
                    % Set parameter value
                    try
                        set(EstimOutObj, obj.EstimOutUIData.Params{Idx}{i,1}, value);
                    catch ME
                        exitFlag = 1;   % Flag an error for calling fxn
                        msgStr = sprintf(['Failed to build EstimOut object ' ...
                            '%s successfully.  Error message: %s'], ...
                            SelectedClassName, ME.message);
                        msgbox(msgStr, 'Error with user-provided input', 'error');
                        return
                    end
                end
            end
        end
        
        
        % *****************************************************************
        %                GENERATE GAMP OPTIONS OBJECT METHOD
        % *****************************************************************
        % This method is called by other public methods of the GUIData
        % class, and is used to create a GampOpt object based on the
        % current state of the GUIData object, obj
        function [GampOptObj, exitFlag] = createGampOptObj(obj)
            exitFlag = 0;   % No error exit flag
            
            GampOptObj = GampOpt();
            % Populate the default object with the values contained in
            % the GAMPOptUIData property
            for i = 1:numel(obj.GAMPOptUIData.GAMPParams(:,1))
                value = obj.GAMPOptUIData.GAMPValues{i,1};   % Current value
                % Set parameter value
                try
                    field = obj.GAMPOptUIData.GAMPParams{i,1};
                    GampOptObj.(field) = value;
                catch ME
                    exitFlag = 1;   % Flag an error for calling fxn
                    msgStr = sprintf(['Failed to build GampOpt object ' ...
                        'successfully.  Error message: %s'], ...
                        ME.message);
                    msgbox(msgStr, 'Error with user-provided input', 'error');
                    return
                end
            end
        end
        
        
        % *****************************************************************
        %                          GENERATEDATA METHOD
        % *****************************************************************
        % This method is called by other public methods of the GUIData
        % class, and is used to create a synthetic dataset
        function [A, y, x_true, exitFlag] = GenerateData(obj, ...
                DataOptObj, EstimInObj, EstimOutObj)
            
            exitFlag = 0;   % Clear error flag
            
            % Extract relevant parameters
            N = DataOptObj.N;
            M = DataOptObj.M;
            Atype = DataOptObj.Atype;
            
            % Generate a synthetic true x of appropriate dimension
            try
                x_true = EstimInObj.genRand(N);
            catch ME
                exitFlag = 1;
                msgStr = sprintf(['Failed to generate true signal ' ...
                    'successfully.  Error message: %s'], ...
                    ME.message);
                msgbox(msgStr, 'Error with data generation', 'error');
                return
            end
            
            % Build a random matrix of the desired type
            switch Atype
                case 'Random Gaussian'
                    a0 = 0;
                    A_mtx = 1/sqrt(N).*(randn(M,N) + a0);
                    A = MatrixLinTrans(A_mtx);
                case 'Complex Gaussian'
                    a0 = 0;
                    A_mtx = 1/sqrt(N).*...
                        (sqrt(1/2)*randn(M,N) + sqrt(1/2)*1j*randn(M,N) + a0);
                    A = MatrixLinTrans(A_mtx);
                case 'Uniform Random'
                    A_mtx = 1/sqrt(N).*rand(M,N);
                    A = MatrixLinTrans(A_mtx);
                case 'Zero-One Random'
                    A_mtx = 1/sqrt(N)*(rand(M,N) > 0.5);
                    A = MatrixLinTrans(A_mtx);
                case 'Sparse Random'
                    d = 10;
                    A_mtx = genSparseMat(M,N,d);
                    A = MatrixLinTrans(A_mtx);
                case 'Row-subsampled DFT'
                    domain = true; %set to false for IDFT
                    A = FourierLinTrans(N,N,domain);
                    A.ySamplesRandom(M); %randomly subsample the DFT matrix
                case 'Block-subsampled DFT'
                    domain = true; %set to false for IDFT
                    A = FourierLinTrans(N,N,domain);
                    start_indx = ceil((N-M)*rand(1));
                    A.ySamplesBlock(M,start_indx); %block-subsample the DFT matrix
                otherwise
                    exitFlag = 1;
                    msgStr = sprintf(['Failed to build LinTrans object' ...
                        'successfully.  Error message: %s'], ...
                        'Unrecognized type');
                    msgbox(msgStr, 'Error with user-provided input', 'error');
                    return
            end
            
            % Build observations, y, by operating on x_true with the
            % LinTrans object, then pass result through EstimOut channel
            % Generate a synthetic true x of appropriate dimension
            try
                y = EstimOutObj.genRand(A.mult(x_true));
            catch ME
                exitFlag = 1;
                msgStr = sprintf(['Failed to construct observations ' ...
                    'successfully.  Error message: %s'], ...
                    ME.message);
                msgbox(msgStr, 'Error with data generation', 'error');
                return
            end
        end
    end
end % classdef