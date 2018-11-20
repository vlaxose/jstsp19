% CLASS: SupportConcat
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: SupportStruct & hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is useful whenever one has constructed a SignalConcat object
%   (see SignalConcat.m) for the purpose of assigning distinct signal
%   priors to different subsets of coefficients of the unknown signal, X.
%   If each subset has an associated distinct support structure, i.e., a
%   distinct SupportStruct object to accompany each Signal object contained
%   in the the SignalConcat object, then this class should be used to
%   aggregate these individual SupportStruct objects within a single
%   SupportConcat container class.
%
% PROPERTIES (State variables)
%   SupportArray    A length-L cell array of SupportStruct objects, where
%                   L is the number of Signal objects contained in the
%                   SignalArray property of the relevant SignalConcat class
%
% METHODS (Subroutines/functions)
%   obj = SupportConcat(SupportArray)
%       - Constructs an object of the SupportConcat class
%   get_type(obj)
%       - Returns the string 'concat' as a character string
%   print(obj)
%    	- Prints the values of the properties of each SupportArray object 
%         to the command window
%   SupportCopyObj = copy(obj)
%       - Creates an independent clone of the SupportConcat object, obj,
%         called SupportCopyObj, i.e., a new SupportConcat object with the 
%         same property values   
% 	[PI_IN, S_POST] = UpdateSupport(obj)
%       - This method is unimplemented since the associated SignalConcat
%         object should distribute each SupportArray element to each
%         SignalArray object, which will then call the UpdateSupport method
%         of the specific SupportStruct object [Hidden method]
%   [S_TRUE] = genRand(TBobj, GenParams)
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

classdef SupportConcat < SupportStruct
    
    properties
        SupportArray;        % A cell array of SupportStruct objects
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
        
        function obj = SupportConcat(SupportArray)
            if ~isa(SupportArray, 'cell')
                error('SupportArray must be a cell array of SupportStruct objects')
            elseif numel(SupportArray) < 2
                error('SupportArray should contain 2 or more SupportStruct objects')
            end
            obj.SupportArray = SupportArray;
        end
        
        
        % *****************************************************************
        %                         GET METHODS
        % *****************************************************************
        
        % Get version of GAMP being run
        function version = get.version(obj)
            ver = obj.SupportArray{1}.version;
            for j = 2:numel(obj.SupportArray)
                if ~strcmpi(ver, obj.SupportArray{j}.version)
                    warning('Mismatch in GAMP version, SupportArray objects')
                end
            end
            version = ver;
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SUPPORT PRIOR: Concatenated SupportStruct array\n')
            fprintf('***********************************************\n')
            for k = 1:numel(obj.SupportArray)
                fprintf('--- SupportArray{%d}  ---\n', k)
                obj.SupportArray{k}.print();
            end
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        
        % Create an indepedent copy of a SupportConcat object
        function SupportConcatCopyObj = copy(obj)
            SuppCopyObj = cell(1,numel(obj.SupportArray));
            for k = 1:numel(obj.SupportArray)
                SuppCopyObj{k} = obj.SupportArray{k}.copy();
            end
            SupportConcatCopyObj = SupportConcat(SuppCopyObj);
        end
        
        
        % *****************************************************************
        %                          DELETE METHOD
        % *****************************************************************
        
%         % Delete the SupportConcat object (and quite likely those objects
%         % that were used to construct it)
%         function delete(obj)
%             for k = 1:numel(obj.SupportArray)
%                 obj.SupportArray{k}.delete();
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
        
        function UpdateSupport(obj)
            % If any Signal object is calling this method for this
            % particular class (but not an inheriting sub-class), throw an
            % exception, as it is expected that the SignalConcat object
            % will parcel out the SupportArray objects to each Signal
            % object within the SignalArray property of the SignalConcat
            % object
            ST = dbstack;
            error(['%s should not call the method %s when support' ...
                ' concatenation is present.'], ...
                ST(2).name, ST(1).name)
        end
        
        
        % *****************************************************************
        %               GENERATE REALIZATION METHOD
        % *****************************************************************
        function genRand(obj, ~, ~)
            % If any Signal object is calling this method for this
            % particular class (but not an inheriting sub-class), throw an
            % exception, as it is expected that the SignalConcat object
            % will parcel out the SupportArray objects to each Signal
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
            ReportTmp = cell(numel(obj.SupportArray), 1);
            % Call EMreport method for each sub-object
            for i = 1:numel(obj.SupportArray)
                ReportTmp{i} = EMreport(obj.SupportArray{i});
            end
            Report = vertcat(ReportTmp{:});
        end
    end
   
end % classdef