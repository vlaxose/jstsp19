% CLASS: NoSupportStruct
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: SupportStruct & hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This concrete class should be used whenever there is no structured
%   sparsity inherent in the signal, or whenever the Signal class object
%   will be handling its own message passing for the support structure
%
% PROPERTIES (State variables)
%   No user-accessible properties for concrete instances of this class
%
% METHODS (Subroutines/functions)
%   NoSupportStruct()
%       - Default constructor.  When there is no structure in the support
%         matrix S, then an object of this class can be constructed using
%         the above syntax, to designate that no structure is present
%   get_type(obj)
%   	- Returns the character string 'None', indicating that no structure
%   	  exists to the support matrix S
%   print(obj)
%   	- Prints the values of the properties to the command window
%   SupportStructCopyObj = copy(obj)
%       - Creates an independent copy of the SupportStruct object, obj
%   [PI_IN, S_POST] = UpdateSupport(obj, TBobj, PI_OUT)
%       - This method should not be called by any Signal class object,
%         since it is assumed that no structure is present or the Signal
%         class object will handle all such message passing [Hidden method]
%   S_TRUE = genRand(obj, TBobj, GenParams)
%       - This method should never be called for this particular class
%         [Hidden method]
%   EMreport(obj)
%       - Returns an empty report (not needed for this class)
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 09/16/12
% Change summary: 
%       - Created from SupportStruct (09/16/12; JAZ)
% Version 0.2
%

classdef NoSupportStruct < SupportStruct
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type = 'None';      % Type of support structure, e.g., 'MC1'
    end % properties
    
    properties
        version = 'mmse';   % Irrelevant parameter
    end
   
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = NoSupportStruct()
            % Default constructor is for no support structure
        end
        
        
        % *****************************************************************
        %                         PRINT METHOD
        % *****************************************************************
        
        function print(obj)
            fprintf('SUPPORT STRUCTURE: None\n')
        end
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        
        % Creates an independent copy of the SupportStruct object
        function SupportStructCopyObj = copy(obj)
            SupportStructCopyObj = NoSupportStruct();
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
        %            	SUPPORT MESSAGES UPDATE METHOD
        % *****************************************************************
        
        % Since there is no structure for concrete objects of the
        % SupportStruct class, Signal class objects should not be calling
        % this method
        function [PI_IN, S_POST] = UpdateSupport(obj)
            
            % If any Signal object is calling this method for this
            % particular class (but not an inheriting sub-class), throw an
            % exception, as it is expected that the Signal object will
            % produce its own updated messages for GAMP in the absence of
            % any structured sparsity
            ST = dbstack;
            error(['%s should not call the method %s when no support' ...
                ' structure is present.  In that case, it should ' ...
                'perform its own updates of GAMP-bound messages.'], ...
                ST(2).name, ST(1).name)
        end
        
        % *****************************************************************
        %            	GENERATE SUPPORT MATRIX REALIZATION
        % *****************************************************************
        
        function S_TRUE = genRand(obj, TBobj, GenParams)
            
            % If any Signal object is calling this method for this
            % particular class, throw an exception, as it is expected that 
            % the Signal object will generate its own realization of the 
            % support matrix in this case
            ST = dbstack;
            error(['%s should not call the method %s when no support' ...
                ' structure is present.  In that case, it should ' ...
                'generate a realization of S itself.'], ...
                ST(2).name, ST(1).name)
        end
        
        
        % *****************************************************************
        %                          EM REPORT METHOD
        % *****************************************************************
        function Report = EMreport(obj)
            % No parameters, thus nothing to report
            Report = [];
        end
    end
   
end % classdef