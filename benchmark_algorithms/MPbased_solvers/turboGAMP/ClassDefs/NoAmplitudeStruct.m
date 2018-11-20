% CLASS: NoAmplitudeStruct
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: AmplitudeStruct & hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used when no amplitude structure is present in the signal
%
% PROPERTIES (State variables)
%   No user-accessible properties for concrete instances of this class
%
% METHODS (Subroutines/functions)
%   get_type(obj)
%    	- Returns 'None' as a character string
%   print(obj)
%    	- Prints the values of the properties to the command window
%   AmplitudeStructCopyObj = copy(obj)
%       - Create an independent copy of an AmplitudeStruct object, obj
%   [varargout] = UpdateAmplitude(obj, TBobj, varargin)
%       - This method should not be called by the Signal class when there
%         is no amplitude structure present or when it is desired that the
%         Signal class object manages its own message computation of 
%         amplitude-related messages [Hidden method]
%   InitPriors(obj, TBobj, Y, A)
%       - This method should not be called by a Signal class object when
%         no structure is present in the amplitudes [Hidden method]
%   EMreport(obj)
%       - Returns an empty report (not needed for this class)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 09/16/12
% Change summary: 
%       - Created from AmplitudeStruct (09/16/12; JAZ)
% Version 0.2
%

classdef NoAmplitudeStruct < AmplitudeStruct
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'None';      % Type of column support structure, e.g., 'GM'
    end % properties
   
    properties
        version = 'mmse';   % Irrelevant parameter
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = NoAmplitudeStruct()
            % Default constructor is for no amplitude structure
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        function print(obj)
            fprintf('AMPLITUDE STRUCTURE: None\n')
        end
        
        
        % *****************************************************************
        %                        COPY METHOD
        % *****************************************************************
        function AmplitudeStructCopyObj = copy(obj)
            AmplitudeStructCopyObj = NoAmplitudeStruct();
        end
        
        
        % *****************************************************************
        %                      ACCESSORY METHOD
        % *****************************************************************
        
        % This function allows one to query which type of column support
        % structure is present
        function type_id = get_type(obj)
            type_id = obj.type;
        end
    end % methods
    
    methods (Hidden)
        % *****************************************************************
        %            	AMPLITUDE MESSAGES UPDATE METHOD
        % *****************************************************************
        
        function [varargout] = UpdateAmplitude(obj, TBobj, varargin)
            % If any Signal object is calling this method for this
            % particular class (but not an inheriting sub-class), throw an
            % exception, as it is expected that the Signal object will
            % produce its own updated messages for GAMP in the absence of
            % any structured sparsity
            ST = dbstack;
            error(['%s should not call the method %s when no amplitude' ...
                ' structure is present.  In that case, it should ' ...
                'perform its own updates of GAMP-bound messages.'], ...
                ST(2).name, ST(1).name)
        end
        
        
        % *****************************************************************
        %            	INITIALIZE MODEL PARAMETERS METHOD
        % *****************************************************************
        
        function InitPriors(obj, TBobj, Y, A)
            % No priors to implement in the case where no amplitude
            % structure is present
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % No parameters, thus nothing to report
            Report = [];
        end
    end %methods
   
end % classdef