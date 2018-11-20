% CLASS: GenParams
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: hgsetget (MATLAB handle class)
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   This class is used to store information needed in order to produce a
%   realization of a signal model that is defined by objects of the Signal,
%   Noise, SupportStruct, and AmplitudeStruct classes.  In all cases, it is
%   assumed that each column of an N-by-T signal matrix, X, whose elements 
%   are generated from a distribution defined by the Signal, SupportStruct,
%   and AmplitudeStruct classes, are transformed according to
%                  z(t) = A(t)*x(t),  t = 1, ..., T
%   where A(t) is an M-by-N linear operator.  These "transform vectors",
%   z(t), are then observed through a separable observation channel defined
%   by the probability distribution p(Y(m,t) | Z(m,t)), (specified by the
%   Noise class object), to yield an M-by-T matrix of observations, Y.
%
%   NOTE: If you wish to specify the SNR (in dB), then assign a value to
%   the property SNRdB.  If no value is assigned to SNRdB, then the noise E
%   will be generated according to the parameters defined in the Noise
%   class object.  If a value is assigned to SNRdB, then this choice will
%   override the parameters of the Noise class object (but data will be
%   generated according to the same noise distribution family).
%
%   To create a GenParams object, there are three constructors to choose
%   from (see METHODS section below).  The default constructor, GenParams(), 
%   will create a GenParams object initialized with all default values for 
%   each parameter/property.  The first alternative constructor allows the 
%   user to initialize any subset of the parameters, with the remaining 
%   parameters initialized to their default values, by using MATLAB's 
%   property/value string pairs convention, e.g.,
%   GenParams('SNRdB', 15, 'M', 64) will construct a GenParams object in 
%   which M = 64, SNRdB = 15, and all remaining parameters are set to their 
%   default values.  The second alternative constructor is to specify every
%   input argument sequentially as follows: GenParams(N, T, M, SNRdB,
%   A_type, type, commonA).
%
% PROPERTIES (State variables)
%   N           Row dimension of X          [Default: 512]
%   T           Column dimension of X       [Default: 4]
%   M           Row dimension of Y          [Default: 128]
%   SNRdB       Per-measurement SNR in dB   [Optional]
%   A_type      Type of random measurement matrix to generate.  (1) for
%               a normalized IID Gaussian matrix, (2) for a Rademacher 
%               matrix, or (3) for subsampled DFT matrix   [Default: 1]
%   type        Type of data to generate ('real' or 'complex') [Default:
%               'real']
%   commonA     Use one transform matrix A for all t = 1, ..., T (true), or
%               use a distinct random A(t) for each t (false) [Default:
%               'true']
%
% METHODS (Subroutines/functions)
%   GenParams()
%       - Default constructor.  Assigns all properties to their default 
%         values.
%   GenParams('ParameterName1', Value1, 'ParameterName2', Value2, ...)
%       - Custom constructor.  Can be used to set any subset of the
%         parameters, with remaining parameters set to their defaults
%   GenParams(N, T, M, SNRdB, A_type, type, commonA)
%       - Full constructor.  Assigns each property to the value given in
%         the appropriate argument
%   set(obj, 'ParameterName', Value)
%       - Method that can be used to set the value of a parameter once the
%         object of class BernGauss, obj, has been constructed
%   print()
%       - Print the current value of each property to the command window
% 

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/03/12
% Change summary: 
%       - Created (01/03/12; JAZ)
% Version 0.2
%

classdef GenParams < hgsetget

    properties
        N = 512;
        T = 4;
        M = 128;
        SNRdB;
        A_type = 1;
        type = 'real';
        commonA = true;
    end
    
    methods
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = GenParams(varargin)
            if nargin == 7 && isa(varargin{1}, 'double')
                % User is using the full constructor
                obj.set('N', varargin{1});
                obj.set('T', varargin{2});
                obj.set('M', varargin{3});
                obj.set('SNRdB', varargin{4});
                obj.set('A_type', varargin{5});
                obj.set('type', varargin{6});
                obj.set('commonA', varargin{7});
            elseif nargin == 1 || mod(nargin, 2) ~= 0
                error('Improper constructor call')
            else
                for i = 1 : 2 : nargin - 1
                    obj.set(varargin{i}, varargin{i+1});
                end
            end
        end                    
        
        
        % *****************************************************************
        %                         SET METHODS
        % *****************************************************************
        
        % Set method for N
        function obj = set.N(obj, N)
            if numel(N) ~= 1
                error('Invalid input: N')
            else
                obj.N = N;
            end
        end
        
        % Set method for T
        function obj = set.T(obj, T)
            if numel(T) ~= 1
                error('Invalid input: T')
            else
                obj.T = T;
            end
        end
        
        % Set method for M
        function obj = set.M(obj, M)
            if numel(M) ~= 1
                error('Invalid input: M')
            else
                obj.M = M;
            end
        end
        
        % Set method for SNRdB
        function obj = set.SNRdB(obj, SNRdB)
            if numel(SNRdB) ~= 1 && numel(SNRdB) ~= 0
                error('Invalid input: SNRdB')
            else
                obj.SNRdB = SNRdB;
            end
        end
        
        % Set method for A_type
        function obj = set.A_type(obj, A_type)
            if A_type ~= 1 && A_type ~= 2 && A_type ~= 3
                error('A_type must equal either 1, 2, or 3')
            else
                obj.A_type = A_type;
            end
        end
        
        % Set method for type
        function obj = set.type(obj, string)
            if sum(strcmpi(string, {'real', 'complex'})) == 0
                error('Invalid option: type')
            else
                obj.type = lower(string);
            end
        end
        
        % Set method for commonA
        function obj = set.commonA(obj, comA)
            if ~isa(comA, 'logical') && comA ~= 1 && comA ~= 0
                error('Invalid option: commonA')
            else
                obj.commonA = logical(comA);
            end
        end
        
        
        % *****************************************************************
        %                        PRINT METHOD
        % *****************************************************************
        
        % Print the current configuration to the command window
        function print(obj)
            fprintf('SIGNAL GENERATION PARAMETERS:\n')
            fprintf('     Signal matrix row dimension (N): %d\n', obj.N)
            fprintf('  Signal matrix column dimension (T): %d\n', obj.T)
            fprintf('Measurement matrix row dimension (M): %d\n', obj.M)
            if isempty(obj.SNRdB)
                fprintf('         Per-measurement SNR (SNRdB): Not assigned\n')
            else
                fprintf('         Per-measurement SNR (SNRdB): %g dB\n', obj.SNRdB)
            end
            switch obj.A_type
                case 1
                    fprintf('                   Meas. matrix type: IID Gaussian\n')
                case 2
                    fprintf('                   Meas. matrix type: Rademacher\n')
                case 3
                    fprintf('                   Meas. matrix type: Subsampled DFT\n')
            end
            switch obj.type
                case 'real'
                    fprintf('                           Data type: Real-valued\n')
                case 'complex'
                    fprintf('                           Data type: Complex-valued\n')
            end
            switch obj.commonA
                case true
                    fprintf('          Single A matrix (A(t) = A): Yes\n')
                case false
                    fprintf('          Single A matrix (A(t) = A): No\n')
            end
        end
        
    end % methods
    
end % classdef