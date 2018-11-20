% CLASS: Signal
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: BernGauss, BernLaplace, ElasticNet, Laplacian, GaussMix, 
%               SignalConcat
% 
% TYPE (Abstract or Concrete)
%   Abstract
%
% DESCRIPTION (High-level overview of the class)
%   This abstract class is used by inheriting sub-classes to define a
%   particular prior distribution family.
%
% PROPERTIES (State variables)
%   version         Character string containing either 'map' or 'mmse', 
%                   indicating the version of GAMP to run (e.g., max-sum
%                   or sum-product, respectively
%   type            The particular type of signal prior family, e.g.,
%                   Bernoulli-Gaussian or Gaussian Mixture
%   data            Character string identifier indicating real-valued
%                   ('real') or complex-valued ('complex') signal
%
% METHODS (Subroutines/functions)
%   get_type(obj)
%       - Returns the value of "type" as a character string
%   print(obj)
%    	- Prints the values of the properties of the Signal object, obj, to 
%         the command window
%   SignalCopyObj = copy(obj)
%       - Creates an independent clone of the Signal class object, obj,
%         called SignalCopyObj, i.e., a new Signal object with the same 
%         property values
% 	[EstimIn, S_POST] = UpdatePriors(TBobj, GAMPState, EstimInOld)
%       - Given the final state of the message passing variables that were 
%         output from GAMP after its most recent execution, produce a new 
%         object of the EstimIn base class that will be used to specify the 
%         signal "prior" on the next iteration of GAMP. TBobj is an object 
%         of the TurboOpt class, GAMPState is an object of the GAMPState 
%         class, and EstimInOld is the previous EstimIn object given to
%         GAMP. If TBobj.commonA is false, then this method should return
%         a 1-by-T cell array of EstimIn objects.  Should also return an
%         estimate of the posterior of the support variables in S_POST.  
%         All derived classes of the Signal class must implement this 
%         method. [Hidden method]
%   EstimIn = InitPriors(TBobj)
%     	- Provides an initial EstimIn object for use by GAMP the first
%         time. This method can be as simple as using the default, or
%         user-provided property values, or as complex as accepting Y and A 
%         as inputs and initializing parameters from the data. TBobj is an 
%         object of the TurboOpt class. If TBobj.commonA is false, then 
%         this method should return a 1-by-T cell array of EstimIn objects.
%         All derived classes of the Signal class must implement this 
%         method. [Hidden method]
%   [X_TRUE, S_TRUE, GAMMA_TRUE] = genRand(TBobj, GenParams)
%       - Using an object of the TurboOpt class (TBobj) and an object of
%         the GenParams class (GenParams), generate a realization of the
%         signal prior (X_TRUE) as well as the underlying support matrix
%         (S_TRUE) and amplitude matrix (GAMMA_TRUE) [Hidden method]
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
%         derived classes of the Signal class must implement this method. 
%         [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (10/11/11; JAZ)
%       - Added UpdatePriors as a mandatory method that concrete subclasses
%         must implement (12/13/11; JAZ)
%       - Added genRand as a mandatory method that concrete subclasses must
%         implement (01/02/12; JAZ)
%       - Added "data" as a required property (05/23/12; JAZ)
%       - Added copy method for creating an independent clone (07/10/12;
%         JAZ)
%       - Added EMreport method (01/28/13; JAZ)
% Version 0.2
%

classdef Signal < hgsetget
    
    properties (Abstract)
        version;    % Character string containing either 'map' or 'mmse', 
                    % indicating the version of GAMP to run (e.g., max-sum
                    % or sum-product, respectively
    end
    
    properties (Abstract, Constant)
        type        % Prior distribution signal family, e.g., BG
    end % properties
    
    properties (Abstract)
        data        % Identifier for real- or complex-valued signal
    end % properties
   
    methods (Abstract)
        
        % *****************************************************************
        %  INHERITING SUB-CLASSES MUST PROVIDE CONCRETE IMPLEMENTATIONS
        %  OF ALL OF THE FOLLOWING METHODS (ALTHOUGH INTERFACES CAN VARY
        %  SOMEWHAT)
        % *****************************************************************
        
        % Use most recent GAMP outputs to produce a new prior distribution
        % for GAMP's next iteration
        EstimIn = UpdatePriors(obj, TBobj, GAMPState, EstimInOld)
        
        % Create an initial EstimIn object for GAMP's first iteration
        EstimIn = InitPriors(obj, TBobj)
        
        % Generate a signal realization
        [X_TRUE, S_TRUE, GAMMA_TRUE] = genRand(TBobj, GenParams)
        
        type_id = get_type(obj)  	% Returns character string for ID
        
        print(obj)  % Prints property values to the command window
        
        % Creates an independent copy of the Signal object
        SignalCopyObj = copy(obj)
        
        % Report on the status of the EM update procedure
        Report = EMreport(obj)
        
    end % methods
   
end % classdef