% CLASS: AmplitudeStruct
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: NoAmplitudeStruct, GaussMarkov, AmplitudeConcat,
%               GaussMarkovFieldArb
% 
% TYPE (Abstract or Concrete)
%   Abstract
%
% DESCRIPTION (High-level overview of the class)
%   This class is used by inheriting subclasses to define a particular form 
%   of structured sparsity in the amplitudes of the non-zero elements of 
%   the N-by-T signal matrix, X
%
% PROPERTIES (State variables)
%   version         Character string containing either 'map' or 'mmse', 
%                   indicating the version of GAMP to run (e.g., max-sum
%                   or sum-product, respectively
%   type            The particular type of structure prior family, e.g.,
%                   Gauss-Markov
%
% METHODS (Subroutines/functions)
%   get_type(obj)
%    	- Returns the value of "type" as a character string
%   print(obj)
%    	- Prints the values of the properties to the command window
%   AmplitudeStructCopyObj = copy(obj)
%       - Create an independent copy of an AmplitudeStruct object, obj
%   [varargout] = UpdateAmplitude(obj, TBobj, varargin)
%       - This method, which is analogous to the SupportStruct method
%         UpdateSupport (see SupportStruct.m), is called by objects of the
%         Signal class as a part of their UpdatePriors method.  The job of
%         this method, which must be implemented by any inheriting
%         sub-classes of this class, is to take messages leaving GAMP and
%         combine them with the particular probabilistic structure of the
%         amplitude variables to generate messages that will create the
%         EstimIn object for the next round of GAMP message passing.  The
%         interface of this method will vary considerably depending on the
%         signal prior, thus no consistent interface can be used for every
%         situation.  For instance, for Bernoulli-Gaussian signal priors,
%         it is expected that outputs would include means and variances of
%         the active component Gaussians.  For a mixture-of-Gaussians 
%         prior, the amplitude structure would consist of tensors of means 
%         and variances, where the third dimension is an index into the 
%         particular Gaussian component.  In addition, UpdateAmplitude 
%         should update any structure-specific applicable parameters, 
%         (e.g., a Gauss-Markov process correlation parameter), that can be 
%         learned using an expectation-maximization (EM) algorithm, if the 
%         user so specifies. [Hidden method]
%   InitPriors(obj, TBobj, Y, A)
%       - This method is called as part of TurboOpt's InitPriors method in
%         order to initialized any applicable model parameters from the
%         data. This method should be implemented by inheriting sub-classes
%         if it is possible for associated model parameters to be 
%         initialized from the data. [Hidden method]
%   Report = EMreport(obj)
%       - This method returns Report, whish is a k-by-3 cell array that
%         summarizes the status of any expectation-maximization (EM) 
%         parameter learning that has taken place on the most recent
%         iteration, where k is the number of parameters with EM learning
%         enabled.  The format of each row of Report is as follows:
%         {'param_name', 'descriptor', value}.  'param_name' is a string
%         that contains the formal name of the parameter being learned
%         (e.g., 'lambda'), 'descriptor' is a string that may be printed to
%         the command window (e.g., 'Exponential decay rate'), and value
%         is a numeric scalar containing the most recent EM update.  All 
%         derived classes of the AmplitudeStruct class must implement this 
%         method. [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (12/10/11; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Made this class abstract (no longer instantiable) to match
%         convention of Signal and Noise classes, moving structureless
%         amplitude class to NoAmplitudeStruct (09/16/12; JAZ)
%       - Added EMreport method (01/28/13; JAZ)
% Version 0.2
%

classdef AmplitudeStruct < hgsetget
    
    properties (Abstract)
        version;    % Character string containing either 'map' or 'mmse', 
                    % indicating the version of GAMP to run (e.g., max-sum
                    % or sum-product, respectively
    end
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type;      % Type of column support structure, e.g., 'GM'
    end % properties
   
    methods (Abstract)
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        print(obj)
            % Print any object property values to the command window
        
        
        % *****************************************************************
        %                        COPY METHOD
        % *****************************************************************
        AmplitudeStructCopyObj = copy(obj)
            % Create an independent copy of the provided object
        
        
        % *****************************************************************
        %                      ACCESSORY METHOD
        % *****************************************************************
        
        % This function allows one to query which type of column support
        % structure is present
        type_id = get_type(obj)
            
    end % methods
    
    methods (Abstract, Hidden)
        % *****************************************************************
        %            	AMPLITUDE MESSAGES UPDATE METHOD
        % *****************************************************************
        % Concrete subclasses should implement this method to return
        % updated amplitude messages to the Signal class object that
        % called them
        [varargout] = UpdateAmplitude(obj, TBobj, varargin)
        
        
        % *****************************************************************
        %            	INITIALIZE MODEL PARAMETERS METHOD
        % *****************************************************************
        % No priors to implement in the case where no amplitude
        % structure is present
        InitPriors(obj, TBobj, Y, A)
        
        
        % *****************************************************************
        %                       EM LEARNING REPORT
        % *****************************************************************
        % Report on the status of the EM update procedure
        Report = EMreport(obj)
        
    end %methods
   
end % classdef