% CLASS: Observation
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: GaussNoise, GaussMixNoise, ProbitChannel, LogitChannel,
%               HingeLossChannel
% 
% TYPE (Abstract or Concrete)
%   Abstract
%
% DESCRIPTION (High-level overview of the class)
%   This abstract class is used by inheriting subclasses to define a
%   particular observation channel model, e.g., additive noise
%
% PROPERTIES (State variables)
%   version         Character string containing either 'map' or 'mmse', 
%                   indicating the version of GAMP to run (e.g., max-sum
%                   or sum-product, respectively
%   type            Identifier of the particular output channel, e.g.,
%                   additive white Gaussian noise ('AWGN')
%   data            Character string identifying if the observations are
%                   real-valued ('real') or complex-valued ('complex')
%
% METHODS (Subroutines/functions)
%   print(obj)
%       - Print the current value of each property to the command window
%   get_type(obj)
%   	- Returns the value of "type" as a character string
%   ObservationCopyObj = copy(obj)
%       - Creates an independent copy of an Observation object, obj
%   EstimOut = UpdatePriors(TBobj, GAMPState, Y, EstimOutOld, A)
%       - Given the final state of the message passing variables that were 
%         output from GAMP after its most recent execution, produce a new 
%         object of the EstimOut base class that will be used to specify
%         the noise "prior" on the next iteration of GAMP. TBobj is an 
%         object of the TurboOpt class, GAMPState is an object of the
%         GAMPState class, and EstimOutOld is the previous EstimOut object
%         given to GAMP. If TBobj.commonA is false, then this method should 
%         return a 1-by-T cell array of EstimOut objects. [Hidden method]
%   EstimOut = InitPriors(TBobj)
%    	- Provides an initial EstimOut object for use by GAMP the first
%         time. This method can be as simple as using the default, or
%         user-provided property values, or as complex as accepting Y and A
%         as inputs and initializing parameters from the data. TBobj is an 
%         object of the TurboOpt class.  If TBobj.commonA is false, then 
%         this method should return a 1-by-T cell array of EstimOut objects
%         [Hidden method]
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
%         derived classes of the Observation class must implement this 
%         method. [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (10/14/11; JAZ)
%       - Added UpdatePriors as a mandatory method that concrete subclasses
%         must implement (12/13/11; JAZ)
%       - Renamed class from Noise to Observation to reflect generality,
%         added "data" as another required property (05/22/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added EMreport method (01/28/13; JAZ)
% Version 0.2
%

classdef Observation < hgsetget
    
    properties (Abstract)
        version;    % Character string containing either 'map' or 'mmse', 
                    % indicating the version of GAMP to run (e.g., max-sum
                    % or sum-product, respectively
    end
    
    properties (Abstract, Constant)
        type;   % Output channel identifier, e.g., 'AWGN'
    end % properties
    
    properties (Abstract)
        data;   % Identifier for real- or complex-valued observations
    end % properties
   
    methods (Abstract)
        
        % *****************************************************************
        %  INHERITING SUB-CLASSES MUST PROVIDE CONCRETE IMPLEMENTATIONS
        %  OF ALL OF THE FOLLOWING METHODS (ALTHOUGH INTERFACES CAN VARY
        %  SOMEWHAT)
        % *****************************************************************
        
        % Generate new EstimOut object using most recent GAMP final state
        EstimOut = UpdatePriors(TBobj, GAMPState, Y, EstimOutOld)
        
        % Create an initial EstimOut object for GAMP's first run
        EstimOut = InitPriors(TBobj)
        
        type_id = get_type(obj)     % Returns character string identifier
        
        print(obj)      % Print values of parameters to command window
        
        % Creates an independent copy of an Observation object
        ObservationCopyObj = copy(obj)
        
        % Report on the status of the EM update procedure
        Report = EMreport(obj)
    end % methods
   
end % classdef