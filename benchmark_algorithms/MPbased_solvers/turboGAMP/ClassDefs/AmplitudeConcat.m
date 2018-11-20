% CLASS: AmplitudeConcat
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: AmplitudeStruct & hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is useful whenever one has constructed a SignalConcat object
%   (see SignalConcat.m) for the purpose of assigning distinct signal
%   priors to different subsets of coefficients of the unknown signal, X.
%   If each subset has an associated distinct amplitude structure, i.e., a
%   distinct AmplitudeStruct object to accompany each Signal object 
%   contained in the the SignalConcat object, then this class should be 
%   used to aggregate these individual AmplitudeStruct objects within a 
%   single AmplitudeConcat container class.
%
% PROPERTIES (State variables)
%   AmplitudeArray	A length-L cell array of AmplitudeStruct objects, where
%                   L is the number of Signal objects contained in the
%                   SignalArray property of the relevant SignalConcat class
%
% METHODS (Subroutines/functions)
%   obj = AmplitudeConcat(AmplitudeArray)
%       - Constructs an object of the AmplitudeConcat class
%   get_type(obj)
%       - Returns the string 'concat' as a character string
%   print(obj)
%    	- Prints the values of the properties of each AmplitudeArray object 
%         to the command window
%   AmplitudeCopyObj = copy(obj)
%       - Creates an independent clone of the AmplitudeConcat object, obj,
%         called AmplitudeCopyObj, i.e., a new AmplitudeConcat object with 
%         the same property values   
% 	[varargout] = UpdateAmplitude(obj, TBobj, varargin)
%       - This method is unimplemented since the associated SignalConcat
%         object should distribute each AmplitudeArray element to each
%         SignalArray object, which will then call the UpdateAmplitude 
%         method of the specific AmplitudeStruct object [Hidden method]
%   InitPriors(obj, TBobj, Y, A)
%       - This method is likewise unimplemented [Hidden method]
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
%       - Created (09/18/12; JAZ)
%       - Added EMreport method implementation (01/31/13; JAZ)
% Version 0.2
%

classdef AmplitudeConcat < AmplitudeStruct
    
    properties
        AmplitudeArray;        % A cell array of Amplitude objects
    end
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'concat';    % Concatenation identifier
    end % properties
    
    properties (Dependent)
        version;            % Sum-product ('mmse') or max-sum ('map') GAMP?
    end
   
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = AmplitudeConcat(AmplitudeArray)
            if ~isa(AmplitudeArray, 'cell')
                error('AmplitudeArray must be a cell array of AmplitudeStruct objects')
            elseif numel(AmplitudeArray) < 2
                error('AmplitudeArray should contain 2 or more AmplitudeStruct objects')
            end
            obj.AmplitudeArray = AmplitudeArray;
        end
        
        
        % *****************************************************************
        %                         GET METHODS
        % *****************************************************************
        
        % Get version of GAMP being run
        function version = get.version(obj)
            ver = obj.AmplitudeArray{1}.version;
            for j = 2:numel(obj.AmplitudeArray)
                if ~strcmpi(ver, obj.AmplitudeArray{j}.version)
                    warning('Mismatch in GAMP version, AmplitudeArray objects')
                end
            end
            version = ver;
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('AMPLITUDE PRIOR: Concatenated AmplitudeStruct array\n')
            fprintf('***********************************************\n')
            for k = 1:numel(obj.AmplitudeArray)
                fprintf('--- AmplitudeArray{%d}  ---\n', k)
                obj.AmplitudeArray{k}.print();
            end
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        
        % Create an indepedent copy of a AmplitudeConcat object
        function AmplitudeConcatCopyObj = copy(obj)
            AmplCopyObj = cell(1,numel(obj.AmplitudeArray));
            for k = 1:numel(obj.AmplitudeArray)
                AmplCopyObj{k} = obj.AmplitudeArray{k}.copy();
            end
            AmplitudeConcatCopyObj = AmplitudeConcat(AmplCopyObj);
        end
        
        
        % *****************************************************************
        %                          DELETE METHOD
        % *****************************************************************
        
%         % Delete the AmplitudeConcat object (and quite likely those objects
%         % that were used to construct it)
%         function delete(obj)
%             for k = 1:numel(obj.AmplitudeArray)
%                 obj.AmplitudeArray{k}.delete();
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
        
        function UpdateAmplitude(~)
            % If any Signal object is calling this method for this
            % particular class (but not an inheriting sub-class), throw an
            % exception, as it is expected that the SignalConcat object
            % will parcel out the AmplitudeArray objects to each Signal
            % object within the SignalArray property of the SignalConcat
            % object
            ST = dbstack;
            error(['%s should not call the method %s when support' ...
                ' concatenation is present.'], ...
                ST(2).name, ST(1).name)
        end
        
        
        % *****************************************************************
        %                   INITIALIZE PRIORS METHOD
        % *****************************************************************
        function InitPriors(~)
            % If any Signal object is calling this method for this
            % particular class (but not an inheriting sub-class), throw an
            % exception, as it is expected that the SignalConcat object
            % will parcel out the AmplitudeArray objects to each Signal
            % object within the SignalArray property of the SignalConcat
            % object
            ST = dbstack;
            error(['%s should not call the method %s when support' ...
                ' concatenation is present.'], ...
                ST(2).name, ST(1).name)
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            ReportTmp = cell(numel(obj.AmplitudeArray), 1);
            % Call EMreport method for each sub-object
            for i = 1:numel(obj.AmplitudeArray)
                ReportTmp{i} = EMreport(obj.AmplitudeArray{i});
            end
            Report = vertcat(ReportTmp{:});
        end
    end
   
end % classdef