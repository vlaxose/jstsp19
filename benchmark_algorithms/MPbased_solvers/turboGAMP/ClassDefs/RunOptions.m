% CLASS: RunOptions
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used to specify various runtime options of EMturboGAMP.
%   These options include numbers of iterations to execute, early 
%   termination tolerances and whether to work silently or verbosely.
%
%   To create an RunOptions object, there are two constructors to choose
%   from (see METHODS section below).  The default constructor, 
%   RunOptions(), will create a RunOptions object initialized with all 
%   default values for each parameter/property.  The alternative 
%   constructor allows the user to initialize any subset of the parameters,
%   with the remaining parameters initialized to their default values, by 
%   using MATLAB's property/value string pairs convention, e.g.,
%   Options('smooth_iters', 10, 'verbose', 'true') will construct a 
%   RunOptions object in which the maximum number of smoothing iterations 
%   is set to 10, and EMturboGAMP will work verbosely.  Any parameters not 
%   explicitly set in the constructor will be set to their default values 
%   (but can be later modified, if desired).  
%
% PROPERTIES (AMP-MMV Configuration Options)
%   smooth_iters    Maximum number of smoothing (turbo) iterations to
%                   perform. [Default: 5]
%   min_iters       Minimum number of smoothing (turbo) iterations to 
%                   perform [Default: 5]
%   verbose         Output information during execution (true), or display
%                   nothing (false) [Default: false]
%   tol             Tolerance for early termination of EMturboGAMP (when
%                   difference between the MMSE estimates of X from current
%                   and previous iterations falls below tol) [Default:
%                   1e-6]
%   warm_start      Warm-start GAMP at each turbo iteration (true) or not
%                   (false) [Default: true]
%
% METHODS (Subroutines/functions)
%   RunOptions()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   RunOptions('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         properties, with remaining properties set to their defaults
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class Options, obj, has been constructed
%   print(obj)
%       - Print the current value of each property to the command window
%   RunOptionsCopyObj = copy(obj)
%       - Creates an independent copy of a RunOptions object, obj
%                   
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 09/18/12
% Change summary: 
%       - Created (02/06/12; JAZ)
%       - Added copy method (07/10/12; JAZ)
%       - Added warm_start Boolean (09/18/12; JAZ)
% Version 0.2
%

classdef RunOptions < hgsetget

    properties
        % Set all parameters to their default values
        smooth_iters = 5;       % Max # of smoothing iterations
        min_iters = 5;          % Min # of smoothing iterations
        verbose = false;        % Work silently
        tol = 1e-6;             % Early termination tolerance
        warm_start = true;      % Warm-start GAMP at each turbo iteration
    end % properties
    
    methods
        % Constructor
        function obj = RunOptions(varargin)
            if nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(lower(varargin{i}), varargin{i+1});
                end
            end
        end                    
        
        % Set method for max # of smoothing iterations
        function obj = set.smooth_iters(obj, smooth_iters)
           if smooth_iters > 0
               obj.smooth_iters = smooth_iters;
               obj.min_iters = min(obj.smooth_iters, obj.min_iters);
           else
              error('Invalid assignment: smooth_iters')
           end
        end
        
        % Set method for min # of smoothing iterations
        function obj = set.min_iters(obj, min_iters)
           if min_iters <= obj.smooth_iters
               obj.min_iters = min_iters;
           else
              obj.min_iters = obj.smooth_iters;
              fprintf(['min_iters exceeded smooth_iters, thus setting ' ...
                  'min_iters equal to smooth_iters\n']);
           end
        end
                
        % Set method for the verbosity
        function obj = set.verbose(obj, verbose)
            obj.verbose = logical(verbose);
        end
        
        % Set method for early termination tolerance
        function obj = set.tol(obj, tol)
           if numel(tol) == 1
               obj.tol = tol;
           else
              error('Invalid assignment: tol')
           end
        end
        
        % Set method for warm-starting Boolean
        function obj = set.warm_start(obj, warm_start)
            obj.warm_start = logical(warm_start);
        end
               
        % Get method for grabbing all the properties at once
        function [smooth_iters, min_iters, verbose, tol, warm_start] = ...
                getOptions(obj)
            smooth_iters = obj.smooth_iters;
            min_iters = obj.min_iters;
            verbose = obj.verbose;
            tol = obj.tol;
            warm_start = obj.warm_start;
        end
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('***********************************\n')
            fprintf('    EMturboGAMP Runtime Options\n')
            fprintf('***********************************\n')
            fprintf('Max. smoothing iterations: %d\n', obj.smooth_iters)
            fprintf('Min. smoothing iterations: %d\n', obj.min_iters)
            switch obj.verbose
                case true
                    fprintf('                Verbosity: Verbose\n')
                case false
                    fprintf('                Verbosity: Silent\n')
            end
            fprintf('   Early termination tol.: %g\n', obj.tol)
            switch obj.warm_start
                case true
                    fprintf('          Warm-start GAMP: Yes\n')
                case false
                    fprintf('          Warm-start GAMP: No\n')
            end
        end
        
        % Create an independent copy of a RunOptions object
        function RunOptionsCopyObj = copy(obj)
            RunOptionsCopyObj = RunOptions('smooth_iters', ...
                obj.smooth_iters, 'min_iters', obj.min_iters, ...
                'verbose', obj.verbose, 'tol', obj.tol, 'warm_start', ...
                obj.warm_start);
        end
          
    end % methods
   
end % classdef