% CLASS: SignalConcat
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: Signal & hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is useful whenever one wishes to concatenate multiple
%   different types of Signal priors together for the purpose of defining
%   distinct priors on different subsets of signal coefficients.  For
%   example, suppose the unknown signal, X, is a column vector of length
%   N1+N2, and that each entry X(n) was Bernoulli-Gaussian for n = 1,...,N1
%   and X(n) was a Bernoulli-Gaussian-mixture for n = N1+1,...,N2.  Then
%   this SignalConcat class can be constructed and used as the Signal
%   property of a TurboOpt object for EMturboGAMP.
%
%   Continuing the example above, to construct the desired SignalConcat
%   object, suppose we have already constructed a BernGauss object, BGobj,
%   that defines the statistics of X(1),...,X(N1), and a GaussMix object,
%   GMobj, that defines the statistics of X(N1+1),...,X(N2).  Then we can
%   construct a SignalConcat object by passing a cell array containing
%   BGobj and GMobj, as well as an array containing the values N1 and N2,
%   e.g., SigCatobj = SignalConcat({BGobj, GMobj}, [N1, N2]);  Note that
%   the ordering of the arguments is important.
%
%   ** At present, this class can only handle vector-valued signals **
%
%   ** If one wishes to apply distinct support- and amplitude-structure
%   models to distinct subsets of coefficients that correspond to distinct
%   Signal objects in the SignalConcat class, then the SupportConcat and
%   AmplitudeConcat objects should be constructed in a similar manner.
%   Otherwise, single support- and amplitude-structure objects will be used
%   across all Signal objects that make up the SignalConcat object. **
%
% PROPERTIES (State variables)
%   SignalArray     A length-L cell array of Signal objects
%   Inds            A length-L+1 array of indices
%
% METHODS (Subroutines/functions)
%   obj = SignalConcat(SignalArray, NumElArray)
%       - Constructs an object of the SignalConcat class, with
%         NumElArray(k) denoting the number of elements of X associated
%         with the Signal object SignalArray{k}
%   get_type(obj)
%       - Returns the string 'concat' as a character string
%   print(obj)
%    	- Prints the values of the properties of each SignalArray object to
%         the command window
%   SignalCopyObj = copy(obj)
%       - Creates an independent clone of the SignalConcat object, obj,
%         called SignalCopyObj, i.e., a new SignalConcat object with the 
%         same property values   
% 	[EstimIn, S_POST] = UpdatePriors(TBobj, GAMPState, EstimInOld)
%       - This method will parcel up the outgoing messages from GAMP to
%         each Signal object in SignalArray as appropriate, calling their
%         respective UpdatePriors methods and receiving EstimIn objects in
%         return, which it will bundle into an EstimInConcat object
%         [Hidden method]
%   EstimIn = InitPriors(TBobj, Y, A)
%     	- This method will call the InitPriors method for each Signal 
%         object in SignalArray to obtain EstimIn objects, which it will
%         then bundle into an EstimConcat object [Hidden method]
%   [X_TRUE, S_TRUE, GAMMA_TRUE] = genRand(TBobj, GenParams)
%       - Using an object of the TurboOpt class (TBobj) and an object of
%         the GenParams class (GenParams), generate a realization of the
%         signal prior (X_TRUE) as well as the underlying support matrix
%         (S_TRUE) and amplitude matrix (GAMMA_TRUE).  Note that the sizing
%         information contained in GenParams must be consistent with that
%         provided when constructing the SignalConcat object 
%         [Hidden method]
%   RemoveIndices(obj, IndexList)
%       - If obj is a SignalConcat object, and IndexList is a boolean
%         vector with a number of elements equal to max(obj.Inds), then
%         this method will adjust obj.Inds to remove those indices, n, for
%         which IndexList[n] = true. [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'active_mean'), 'descriptor' is a string that may be 
%         printed to the command window (e.g., 'Gaussian mean'), and 
%         value is a numeric scalar containing the most recent EM update. 
%         [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/31/13
% Change summary: 
%       - Created (09/16/12; JAZ)
%       - Added EMreport method implementation (01/31/13; JAZ)
% Version 0.2
%

classdef SignalConcat < Signal
    
    properties
        SignalArray;        % A cell array of Signal objects
        Inds;               % An array that specifies which indices of X 
                            % are associated with each cell of SignalArray
    end
    
    properties (Constant, Hidden)
        type = 'concat';    % Concatenation identifier
    end % properties
    
    properties (Dependent = true)
        data;       % Identifier for real- or complex-valued signal
        version;    % Sum-product ('mmse') or max-sum ('map') GAMP?
        NumCat;     % Number of Signal objects being concatenated
    end % properties
   
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = SignalConcat(SignalArray, NumElArray)
            if nargin == 2
                if ~isa(SignalArray, 'cell')
                    error('SignalArray must be a cell array of Signal objects')
                end
                L = numel(SignalArray);     % # of Signal objects to concat
                obj.Inds = NaN(1,L+1);
                obj.SignalArray = cell(1,L);
                obj.Inds(1) = 1;
                for l = 1:L
                    if ~isa(SignalArray{l}, 'Signal')
                        error(['SignalArray must be a cell array of ' ...
                            'Signal objects'])
                    else
                        obj.SignalArray{l} = SignalArray{l};
                    end
                    obj.Inds(l+1) = obj.Inds(l) + NumElArray(l);
                end
            else
                error('SignalConcat constructor requires 2 arguments')
            end
        end                    
        
        
        % *****************************************************************
        %                         GET METHODS
        % *****************************************************************
        
        % Get whether data is real or complex
        function data = get.data(obj)
            data = obj.SignalArray{1}.data;
        end
        
        % Get number of Signal objects being concatenated
        function NumCat = get.NumCat(obj)
            NumCat = numel(obj.SignalArray);
        end
        
        % Get version of GAMP being run
        function version = get.version(obj)
            ver = obj.SignalArray{1}.version;
            for j = 2:numel(obj.SignalArray)
                if ~strcmpi(ver, obj.SignalArray{j}.version)
                    warning('Mismatch in GAMP version, SignalArray objects')
                end
            end
            version = ver;
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SIGNAL PRIOR: Concatenated Signal array\n')
            fprintf('***************************************\n')
            for k = 1:obj.NumCat
                fprintf('--- SignalArray{%d}  ---\n', k)
                obj.SignalArray{k}.print();
            end
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        
        % Create an indepedent copy of a SignalConcat object
        function SignalConcatCopyObj = copy(obj)
            NumElArray = NaN(1,obj.NumCat);
            SigArrayCopy = cell(1,obj.NumCat);
            for k = 1:obj.NumCat
                NumElArray(k) = obj.Inds(k+1) - obj.Inds(k);
                SigArrayCopy{k} = obj.SignalArray{k}.copy();
            end
            SignalConcatCopyObj = SignalConcat(SigArrayCopy, NumElArray);
        end
        
        
        % *****************************************************************
        %                          DELETE METHOD
        % *****************************************************************
        
        % Delete the SignalConcat object (and quite likely those objects
        % that were used to construct it)
%         function delete(obj)
%             for k = 1:obj.NumCat
%                 obj.SignalArray{k}.delete();
%             end
%         end
        
        
        % *****************************************************************
        %                      ACCESSORY METHOD
        % *****************************************************************
        
        % This function allows one to query which type of signal family
        % this class is by returning a character string type identifier
        function type_id = get_type(obj)
            type_id = obj.type;
        end            
    end % methods
    
    methods (Hidden)
        % *****************************************************************
        %               UPDATE GAMP SIGNAL "PRIOR" METHOD
        % *****************************************************************
        
        function [EstimIn, S_POST] = UpdatePriors(obj, TBobj, ...
                GAMPStateObj, EstimInOld)
            
            % Unpack the GAMPState object
            [xhat, xvar, rhat, rvar] = GAMPStateObj.getState();
            
            % Allocate space for the new EstimIn object, which will be an
            % EstimInConcat object and for S_POST
            EstimInArray = cell(1,obj.NumCat);
            S_POST = NaN(obj.Inds(end)-1,1);
            
            % Now we must partition up the GAMP outputs according to the
            % indices of X that belong to each Signal object in
            % SignalArray.  Furthermore, since custom forms of support and
            % amplitude structure might be present for each Signal object,
            % we must also break apart the TBobj object into constituent
            % components if its SupportStruct and AmplitudeStruct
            % properties are concatenations.
            for k = 1:obj.NumCat
                % Indices belonging to this Signal object
                SubInd = (obj.Inds(k) : obj.Inds(k+1) - 1)';
                
                % Extract the relevant outgoing GAMP variables
                SubGAMPState = GAMPState(xhat(SubInd), xvar(SubInd), ...
                    rhat(SubInd), rvar(SubInd), [], [], [], []);
                
                % Extract the relevant old EstimIn object
                SubEstimIn = EstimInOld.estimArray{k};
                
                % Build an extracted TurboOpt object that will have
                % appropriate SupportStruct and AmplitudeStruct properties,
                % if they are themselves concatenations
                if isa(TBobj.SupportStruct, 'SupportConcat')
                    % Get the sub-SupportStruct object
                    subSuppObj = TBobj.SupportStruct.SupportArray{k};
                else
                    subSuppObj = TBobj.SupportStruct;
                end
                if isa(TBobj.AmplitudeStruct, 'AmplitudeConcat')
                    % Get the sub-AmplitudeStruct object
                    subAmplObj = TBobj.AmplitudeStruct.AmplitudeArray{k};
                else
                    subAmplObj = TBobj.AmplitudeStruct;
                end
                SubTBobj = TurboOpt('Signal', obj.SignalArray{k}.copy(), ...
                    'SupportStruct', subSuppObj.copy(), 'AmplitudeStruct', ...
                    subAmplObj.copy(), 'Observation', TBobj.Observation.copy(), ...
                    'RunOptions', TBobj.RunOptions.copy());
                
                % Finally, call the UpdatePriors method for this Signal
                % object
                [subEstimIn, subSPOST] = UpdatePriors(obj.SignalArray{k}, ...
                    SubTBobj, SubGAMPState, SubEstimIn);
                
                % Place outputs into permanent location
                EstimInArray{k} = subEstimIn;
                S_POST(SubInd) = subSPOST;
                
                % Copy any parameter learning that took place back to the
                % calling TBobj
%                 TBobj.Signal.SignalArray{k} = SubTBobj.Signal.copy();
                if isa(TBobj.SupportStruct, 'SupportConcat')
                    % Get the sub-SupportStruct object
                    TBobj.SupportStruct.SupportArray{k} = ...
                        SubTBobj.SupportStruct.copy();
                else
                    TBobj.SupportStruct = SubTBobj.SupportStruct.copy();
                end
                if isa(TBobj.AmplitudeStruct, 'AmplitudeConcat')
                    % Get the sub-AmplitudeStruct object
                    TBobj.AmplitudeStruct.AmplitudeArray{k} = ...
                        SubTBobj.AmplitudeStruct.copy();
                else
                    TBobj.AmplitudeStruct = SubTBobj.AmplitudeStruct.copy();
                end
            end
            
            % Now build the outgoing EstimInConcat object
            NumElArray = NaN(1,obj.NumCat);
            for k = 1:obj.NumCat
                NumElArray(k) = obj.Inds(k+1) - obj.Inds(k);
            end
            EstimIn = EstimInConcat(EstimInArray, NumElArray);
            
        end
        
        
        % *****************************************************************
        %         	   INITIALIZE GAMP SIGNAL "PRIOR" METHOD
        % *****************************************************************
        
        function EstimIn = InitPriors(obj, TBobj, Y, A)
                        
            % Allocate space for the new EstimIn object, which will be an
            % EstimInConcat object and for S_POST
            EstimInArray = cell(1,obj.NumCat);
            
            % Find out if A is a MatrixLinTrans operator
            isMtxLinTrans = isa(A, 'MatrixLinTrans');
            
            % Now, since custom forms of support and amplitude structure 
            % might be present for each Signal object, we must break apart 
            % the TBobj object into constituent components if its 
            % SupportStruct and AmplitudeStruct properties are 
            % concatenations.
            for k = 1:obj.NumCat
                % Indices belonging to this Signal object
                SubInd = obj.Inds(k) : obj.Inds(k+1) - 1;
            
                % Build an extracted TurboOpt object that will have
                % appropriate SupportStruct and AmplitudeStruct properties,
                % if they are themselves concatenations
                if isa(TBobj.SupportStruct, 'SupportConcat')
                    % Get the sub-SupportStruct object
                    subSuppObj = TBobj.SupportStruct.SupportArray{k};
                else
                    subSuppObj = TBobj.SupportStruct;
                end
                if isa(TBobj.AmplitudeStruct, 'AmplitudeConcat')
                    % Get the sub-SupportStruct object
                    subAmplObj = TBobj.AmplitudeStruct.AmplitudeArray{k};
                else
                    subAmplObj = TBobj.AmplitudeStruct;
                end
                SubTBobj = TurboOpt('Signal', obj.SignalArray{k}.copy(), ...
                    'SupportStruct', subSuppObj.copy(), 'AmplitudeStruct', ...
                    subAmplObj.copy(), 'Observation', TBobj.Observation.copy(), ...
                    'RunOptions', TBobj.RunOptions.copy());
                
                % If the linear transform A is a matrix, we can try and
                % extract the columns that belong just to this particular
                % Signal object.  Otherwise, just leave as is
                if isMtxLinTrans
                    mtxA = A.A;
                    mtxAvar = A.Avar;
                    if isscalar(mtxAvar)
                        subA = MatrixLinTrans(mtxA(:,SubInd), mtxAvar);
                    else
                        subA = MatrixLinTrans(mtxA(:,SubInd), ...
                            mtxAvar(:,SubInd));
                    end
                else
                    subA = A;
                end
                
                % Finally, call the InitPriors method for this Signal
                % object
                subEstimIn = obj.SignalArray{k}.InitPriors(SubTBobj, Y, ...
                    subA);
                
                % Place outputs into permanent location
                EstimInArray{k} = subEstimIn;
            end
            
            % Now build the outgoing EstimInConcat object
            NumElArray = NaN(1,obj.NumCat);
            for k = 1:obj.NumCat
                NumElArray(k) = obj.Inds(k+1) - obj.Inds(k);
            end
            EstimIn = EstimInConcat(EstimInArray, NumElArray);
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        % Call this method to generate a realization of the
        % Bernoulli-Gaussian prior on X
        %
        % INPUTS:
        % obj       	An object of the BernGauss class
        % TBobj         An object of the TurboOpt class
        % GenParams 	An object of the GenParams class
        %
        % OUTPUTS:
        % X_TRUE        A realization of the signal matrix, X
        % S_TRUE        A realization of the support matrix, S
        % GAMMA_TRUE    A realization of the amplitude matrix, GAMMA
        function [X_TRUE, S_TRUE, GAMMA_TRUE] = genRand(obj, TBobj, GenParams)
            % Generate the signal in pieces.  Start by allocating space for
            % each output variable
            X_TRUE = NaN(obj.Inds(end)-1,1);
            S_TRUE = NaN(obj.Inds(end)-1,1);
            GAMMA_TRUE = NaN(obj.Inds(end)-1,1);
            
            for k = 1:obj.NumCat
                % Indices belonging to this Signal object
                SubInd = (obj.Inds(k) : obj.Inds(k+1) - 1)';
            
                % Build an extracted TurboOpt object that will have
                % appropriate SupportStruct and AmplitudeStruct properties,
                % if they are themselves concatenations
                if isa(TBobj.SupportStruct, 'SupportStructConcat')
                    % Get the sub-SupportStruct object
                    subSuppObj = TBobj.SupportStruct.SupportArray{k};
                else
                    subSuppObj = TBobj.SupportStruct;
                end
                if isa(TBobj.AmplitudeStruct, 'AmplitudeStructConcat')
                    % Get the sub-SupportStruct object
                    subAmplObj = TBobj.AmplitudeStruct.AmplitudeArray{k};
                else
                    subAmplObj = TBobj.AmplitudeStruct;
                end
                SubTBobj = TurboOpt('Signal', obj.SignalArray{k}, ...
                    'SupportStruct', subSuppObj, 'AmplitudeStruct', ...
                    subAmplObj, 'Observation', TBobj.Observation, ...
                    'RunOptions', TBobj.RunOptions);
                
                % Now call the genRand method for this Signal object
                [X_TRUE(SubInd), S_TRUE(SubInd), GAMMA_TRUE(SubInd)] = ...
                    obj.SignalArray{k}.genRand(SubTBobj, GenParams);
            end
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            ReportTmp = cell(obj.NumCat, 1);
            % Call EMreport method for each sub-object
            for i = 1:numel(obj.SignalArray)
                ReportTmp{i} = EMreport(obj.SignalArray{i});
            end
            Report = vertcat(ReportTmp{:});
        end
        
        
        % *****************************************************************
        %                       REMOVE INDICES METHOD
        % *****************************************************************
        % Update the obj.Inds array to account for the removal of certain
        % indices (those for which IndexList[n] = true) from the signal
        %
        % INPUTS:
        % obj       	An object of the BernGauss class
        % IndexList     A boolean array of length max(obj.Inds), with true
        %               values at those locations which are to removed
        %
        function RemoveIndices(obj, IndexList)
            NumRemoved = 0;
            NewIndsArray = [1, NaN(1, obj.NumCat)];     % Temp Inds array
            for k = 1:obj.NumCat
                % Calculate number of indices removed from this subset of
                % coefficients
                NumRemoved = NumRemoved + sum(IndexList(obj.Inds(k) : ...
                    obj.Inds(k+1) - 1));
                NewIndsArray(k+1) = obj.Inds(k+1) - NumRemoved;
                if isa(obj.SignalArray{k}, 'SignalConcat')
                    % Repeat the removal procedure for nested Signal Concat
                    % objects
                    RemoveIndices(obj.SignalArray{k}, IndexList(obj.Inds(k) : ...
                    obj.Inds(k+1) - 1));
                end
            end
            obj.Inds = NewIndsArray;    % Copy temp array to object
        end
    end
   
end % classdef