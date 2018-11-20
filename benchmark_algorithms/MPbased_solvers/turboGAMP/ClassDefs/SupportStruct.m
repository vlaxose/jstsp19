% CLASS: SupportStruct
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: NoSupportStruct, JointSparse, MarkovChain1, MarkovChainD, 
%               MarkovField, MarkovField3D, SupportConcat, MarkovFieldArb
% 
% TYPE (Abstract or Concrete)
%   Abstract
%
% DESCRIPTION (High-level overview of the class)
%   This abstract class is used by inheriting subclasses to define a
%   particular form of structured sparsity in the support of the N-by-T 
%   signal matrix, X.  The support is assumed stored in an N-by-T matrix S, 
%   whose elements can assume D+1 discrete values, {0, 1, ..., D}.
%   Typically, the value 0 corresponds to an inactive element of X, while
%   the values 1, ..., D index specific active distributions.
%
% PROPERTIES (State variables)
%   version         Character string containing either 'map' or 'mmse', 
%                   indicating the version of GAMP to run (e.g., max-sum
%                   or sum-product, respectively
%   type            The particular type of structure prior family, e.g.,
%                   JointSparse
%
% METHODS (Subroutines/functions)
%   SupportStruct()
%       - Default constructor.  When there is no structure in the support
%         matrix S, then an object of this class can be constructed using
%         the above syntax, to designate that no structure is present
%   get_type(obj)
%   	- Returns a simple character string, e.g., 'MC1' for a 1-D Markov
%   	  chain, to indicate the type of structure model
%   print(obj)
%   	- Prints the values of any class properties to the command window
%   SupportStructCopyObj = copy(obj)
%       - Creates an independent copy of the SupportStruct object, obj
%   [PI_IN, S_POST] = UpdateSupport(obj, TBobj, PI_OUT)
%       - This method must be implemented by any sub-class of the
%         SupportStruct class.  The function accepts as inputs an object of
%         the TurboOpt class, TBobj, and PI_OUT, an N-by-T-by-D tensor
%         of outgoing messages to the S variable nodes from GAMP's last
%         iteration.  Using these inputs, UpdateSupport will produce two
%         outputs, PI_IN and S_POST.  PI_IN is an N-by-T-by-D tensor
%         that consists of the messages that will be passed from the S
%         variable nodes to the factor nodes connected to X.  Note that the
%         third dimension of the tensor is D, and not D+1, (see DESCRIPTION
%         above for the definition of D).  This is because the "missing"
%         element of PI_IN(n,t), which corresponds to the message related 
%         to S(n,t) = 0, can be implicitly obtained from the D remaining
%         messages, since all D+1 messages for each n,t, must sum to 1 in
%         order to form a valid pmf.  So, the message related to S(n,t) =
%         d, d = 1, ..., D, is given by PI_IN(n,t,d).  Likewise, S_POST
%         is an N-by-T-by-D tensor of marginal posteriors, i.e.,
%         Pr{S(n,t) = d | Y} = S_POST(n,t,d), for d = 1, ..., D, while
%         Pr{S(n,t) = 0 | Y} = 1 - sum(S_POST(n,t,:)).  In addition to
%         returning these variables, UpdateSupport should perform EM
%         learning of any applicable properties of the SupportStruct
%         derived class, and of the Signal class object that called this
%         method, and update those parameter values in the TBobj object
%         if the user has specified to do so. [Hidden method]
%   S_TRUE = genRand(obj, TBobj, GenParams)
%       - Given a TurboOpt object (TBobj) and a GenParams object, generate
%         a realization of the structure sparse signal support matrix,
%         S_TRUE [Hidden method]
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
%         derived classes of the SupportStruct class must implement this 
%         method. [Hidden method]
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/28/13
% Change summary: 
%       - Created (10/21/11; JAZ)
%       - Removed the ColSuppStruct and RowSuppStruct concrete subclasses,
%         and made SupportStruct a concrete, instantiable class (12/09/11;
%         JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Made this class abstract (no longer instantiable) to match
%         convention of Signal and Noise classes, moving structureless
%         support class to NoSupportStruct (09/16/12; JAZ)
%       - Added EMreport method (01/28/13; JAZ)
% Version 0.2
%

classdef SupportStruct < hgsetget
    
    properties (Abstract)
        version;    % Character string containing either 'map' or 'mmse', 
                    % indicating the version of GAMP to run (e.g., max-sum
                    % or sum-product, respectively
    end
    
    properties (GetAccess = private, SetAccess = immutable, Hidden)
        type;      % Type of support structure, e.g., 'MC1'
    end % properties
   
    methods (Abstract)
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        % Constructs an object of the class
        
        
        % *****************************************************************
        %                         PRINT METHOD
        % *****************************************************************
        % Prints any object property values to the command window
        print(obj)
        
        
        % *****************************************************************
        %                          COPY METHOD
        % *****************************************************************
        % Creates an independent copy of the SupportStruct object
        SupportStructCopyObj = copy(obj)
        
        
        % *****************************************************************
        %                      ACCESSORY METHODS
        % *****************************************************************
        % This function allows one to query which type of column support
        % structure is present
        type_id = get_type(obj)
        
        
    end % methods
    
    methods (Abstract, Hidden)
        % *****************************************************************
        %            	SUPPORT MESSAGES UPDATE METHOD
        % *****************************************************************
        % Using outbound messages from GAMP, provide updated messages
        % for the support variables
        [PI_IN, S_POST] = UpdateSupport(obj)
        
        
        % *****************************************************************
        %            	GENERATE SUPPORT MATRIX REALIZATION
        % *****************************************************************
        % Generate a realization of the support acording to the
        % particular form of structure indicated by the specific
        % implemented SupportStruct class
        S_TRUE = genRand(obj, TBobj, GenParams)
        
        
        % *****************************************************************
        %                       EM LEARNING REPORT
        % *****************************************************************
        % Report on the status of the EM update procedure
        Report = EMreport(obj)
    end
   
end % classdef