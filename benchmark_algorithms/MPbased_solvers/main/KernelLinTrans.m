% CLASS: KernelLinTrans
% 
% HIERARCHY (Enumeration of the various super- and subclasses)
%   Superclasses: LinTrans, hgsetget
%   Subclasses: N/A
% 
% TYPE (Abstract or Concrete)
%   Concrete
%
% DESCRIPTION (High-level overview of the class)
%   The KernelLinTrans class works by constructing an M-by-M "kernel
%   matrix," A_kern, from an M-by-N "feature matrix," A_feat, that is 
%   provided during construction.  Once the kernel matrix has been 
%   constructed, it performs straightforward matrix multiplication against 
%   length-M vectors provided as arguments to any of the mult* methods 
%   described below.  The kernel matrix is constructed such that the
%   (i,j)th entry of A_kern is given by
%               A_kern(i,j) = K(A_feat(i,:), A_feat(j,:)),
%   where the kernel function, K(a, b), represents an inner product between
%   vectors a and b in a (usually) different vector space.  The kernel
%   function must satisfy Mercer's theorem, i.e., a symmetric function that
%   yields a positive semi-definite kernel matrix when applied to any
%   finite set of samples from the domain of the function.
%
%   Commonly used kernels include:
%       - Linear: K(a, b) = <a, b> = a'*b
%       - Gaussian: K(a, b) = exp(-gamma * norm(a - b)^2)
%       - Polynomial: K(a, b) = (<a,b> + offset)^degree
%
% PROPERTIES (State variables)
%   A_feat      An M-by-N matrix of features that will be used to construct
%               the kernel matrix, A_kern, according to the procedure
%               described in DESCRIPTION
%   kernel      Either a string identifier to a pre-defined kernel function
%               (i.e., 'linear', 'gaussian', or 'polynomial') or a function
%               handle of the form @(Ai,Aj) kernel_fxn(Ai,Aj) that behaves
%               according to the function handle convention used by
%               Matlab's 'pdist' command (check the pdist documentation
%               for the DISTFUN example) and returns values of the 
%               kernel function [Default: 'gaussian']
%   gamma       The value of gamma in the Gaussian kernel, if a Gaussian
%               kernel is specified (see DESCRIPTION) [Default: 0.5]
%   offset      The value of the offset term in the polynomial kernel, if a
%               polynomial kernel is specified (see DESCRIPTION) [Default:
%               1]
%   degree      The degree of the polynomial in the polynomial kernel, if a
%               polynomial kernel is specified (see DESCRIPTION) [Default:
%               2]
%   A_kern      The kernel matrix constructed from A_feat and the chosen
%               kernel function (see DESCRIPTION)
%
% METHODS (Subroutines/functions)
%   KernelLinTrans(A_feat, kernel, gamma, offset, degree)
%       - Default constructor.  If any subset of arguments is empty or
%         missing (with the exception of A_feat), the corresponding
%         properties will be set to their default values.
%   B_kern = feat2kern(obj, B_feat)
%       - Given an object of the KernelLinTrans class, obj, and an L-by-N
%         matrix, B_feat, whose rows consist of vectors from the same 
%         feature space as A_feat, feat2kern will return the result of
%         computing the kernel inner product of each feature vector in
%         B_feat against every feature vector in A_feat in the L-by-M
%         matrix of kernel-space vectors, B_kern.  This function is
%         useful when, e.g., one has training and test matrices in a
%         classification problem, and one wishes to train a classifier on a
%         kernel matrix, A_kern, instead of the training feature matrix,
%         A_feat.  Then, to perform prediction on the test set, B_feat, one
%         uses feat2kern to project B_feat into the kernel space, and
%         performs prediction using the new set of "kernel features",
%         B_kern
%   [KLT_train, B_kern_test] = crossvalPartition(obj, partition, foldID)
%       - Given an object of the KernelLinTrans class, obj, and a length-M
%         vector, partition, which is used to define a partition of the M
%         training examples of A_feat into distinct folds for the purposes
%         of cross-validation (type "help crossvalind" to understand the
%         form of the vector partition), this method will return a new
%         KernelLinTrans object, KLT_train, formed from a subset of the
%         full training feature matrix, A_train, consisting of all folds
%         except for the fold specified by the index foldID (i.e., an
%         integer from 1 to k, where k is the total # of unique folds).
%         The method will also return, in B_kern_test, a matrix that is the
%         result of applying the feat2kern method on the left-out fold of
%         the training data (foldID).
%   [M, M] = size(obj)
%       - Returns the size of the implicit A operator, i.e., A_kern
%   z = mult(obj, x)
%       - Perform the forward matrix multiplication, A_kern*x
%   x = multTr(obj, z)
%       - Perform the adjoint matrix multiplication, A_kern'*x
%   z = multSq(obj, x)
%       - Perform the elementwise-squared multiplication, (A_kern.^2)*x
%   x = multSqTr(obj, z)
%       - Perform the elementwise-squared adjoint multiplication, 
%         (A_kern.^2)'*x
%

%
% Coded by: Justin Ziniel, The Ohio State Univ.
% E-mail: zinielj@ece.osu.edu
% Last change: 01/25/13
% Change summary: 
%       - Created (12/07/12; JAZ)
% Version 0.1
%

classdef KernelLinTrans < LinTrans & hgsetget
        
    properties (Dependent)
        A_feat;                 % M-by-N feature matrix used to make A_kern
        kernel;                 % Default to a Gaussian kernel
        gamma;                  % Gaussian kernel scale
        offset;                 % Polynomial kernel offset
        degree;                 % Polynomial kernel degree
    end
    
    properties (Access = private)
        Private_A_feat;         % Actual storage location for A_feat
        Private_kernel = 'gaussian';    % Kernel choice
        Private_gamma = 0.5;    % Gaussian kernel scale actual storage
        Private_offset = 1;     % Polynomial kernel offset actual storage
        Private_degree = 2;     % Polynomial kernel degree actual storage
        kernel_fxn_handle;      % Holds appropriate kernel function handle
        recalc_kern_mtx = true; % Flag indicating that kernel matrix must
                                % be recomputed due to a change in one of
                                % the properties
    end
    
    properties (SetAccess = private);
        A_kern;                 % M-by-M kernel matrix
        A_kern_sq;             	% A_kern.^2
    end
    
    methods
        
        % *****************************************************************
        %                      CONSTRUCTOR METHOD
        % *****************************************************************
        
        function obj = KernelLinTrans(A_feat, kernel, gamma, offset, degree)
            obj = obj@LinTrans;
            if nargin == 0
                % User hasn't assigned any parameters
            else
                obj.A_feat = A_feat;
                
                % Specify a default kernel function handle (Gaussian)
                obj.kernel_fxn_handle = ...
                    @(a,B) exp(-obj.Private_gamma * ...
                    sum(abs(repmat(a, size(B,1), 1) - B).^2, 2));
            end
            if nargin >= 2 && ~isempty(kernel)
                obj.kernel = kernel;
            end
            if nargin >= 3 && ~isempty(gamma)
                obj.gamma = gamma;
            end
            if nargin >= 4 && ~isempty(offset)
                obj.offset = offset;
            end
            if nargin >= 5 && ~isempty(degree)
                obj.degree = degree;
            end
        end
        
        
        % *****************************************************************
        %                         SET METHODS
        % *****************************************************************
        
        % Feature matrix assignment
        function set.A_feat(obj, A_feat)
            if ismatrix(A_feat)
                obj.Private_A_feat = A_feat;
                obj.recalc_kern_mtx = true; % Set recalc flag
            elseif isa(A_feat, 'MatrixLinTrans')
                obj.Private_A_feat = A_feat.A;
                obj.recalc_kern_mtx = true; % Set recalc flag
            else
                error('A_feat must be an explicit matrix')
            end
        end
        
        % Kernel assignment
        function set.kernel(obj, kernel)
            if ischar(kernel)
                % Predefined kernel chosen
                switch lower(kernel)
                    case 'linear'
                        obj.Private_kernel = 'linear';
                        obj.kernel_fxn_handle = @(a,B) (a*(B.')).';
                    case 'gaussian'
                        obj.Private_kernel = 'gaussian';
                        obj.kernel_fxn_handle = ...
                            @(a,B) exp(-obj.Private_gamma * ...
                            sum(abs(repmat(a, size(B,1), 1) - B).^2, 2));
                    case 'polynomial'
                        obj.Private_kernel = 'polynomial';
                        obj.kernel_fxn_handle = ...
                            @(a,b) ((a*(B.')).' + obj.Private_offset).^ ...
                            obj.Private_degree;
                    otherwise
                        error('Unrecognized kernel!')
                end
            elseif isa(kernel, 'function_handle')
                obj.Private_kernel = kernel;
                obj.kernel_fxn_handle = kernel;
            else
                error(['''kernel'' must either be a character string ' ...
                    'or a function handle']);
            end
            obj.recalc_kern_mtx = true; % Set recalc flag
        end
        
        % Gaussian kernel gamma assignment
        function set.gamma(obj, gamma)
            if isnumeric(gamma) && isscalar(gamma) && gamma > 0
                obj.Private_gamma = gamma;
                if ischar(obj.Private_kernel) && strcmpi('gaussian', ...
                        obj.Private_kernel)
                    obj.kernel_fxn_handle = ...
                        @(a,B) exp(-obj.Private_gamma * ...
                        sum(abs(repmat(a, size(B,1), 1) - B).^2, 2));
                end
                obj.recalc_kern_mtx = true;     % Set recalc flag
            else
                error('''gamma'' must be non-negative scalar')
            end
        end
        
        % Polynomial offset assignment
        function set.offset(obj, offset)
            if isnumeric(offset) && isscalar(offset)
                obj.Private_offset = offset;
                if ischar(obj.Private_kernel) && strcmpi('polynomial', ...
                        obj.Private_kernel)
                    obj.kernel_fxn_handle = ...
                        @(a,b) ((a*(B.')).' + obj.Private_offset).^ ...
                        obj.Private_degree;
                end
                obj.recalc_kern_mtx = true;     % Set recalc flag
            else
                error('''offset'' must be a scalar')
            end
        end
        
        % Polynomial degree assignment
        function set.degree(obj, degree)
            if isnumeric(degree) && isscalar(degree) && degree > 0
                obj.Private_degree = degree;
                if ischar(obj.Private_kernel) && strcmpi('polynomial', ...
                        obj.Private_kernel)
                    obj.kernel_fxn_handle = ...
                        @(a,b) ((a*(B.')).' + obj.Private_offset).^ ...
                        obj.Private_degree;
                end
                obj.recalc_kern_mtx = true;     % Set recalc flag
            else
                error('''degree'' must be non-negative scalar')
            end
        end
        
        % Flag for recalculating the kernel matrix
        function set.recalc_kern_mtx(obj, val)
            if islogical(val)
                obj.recalc_kern_mtx = val;
            end
        end
        
        
        % *****************************************************************
        %                           GET METHODS
        % *****************************************************************
        
        % Get feature matrix
        function A_feat = get.A_feat(obj)
            A_feat = obj.Private_A_feat;
        end
        
        % Get kernel
        function kernel = get.kernel(obj)
            kernel = obj.Private_kernel;
        end
        
        % Get Gaussian scale
        function gamma = get.gamma(obj)
            gamma = obj.Private_gamma;
        end
        
        % Get polynomial offset
        function offset = get.offset(obj)
            offset = obj.Private_offset;
        end
        
        % Get polynomial degree
        function degree = get.degree(obj)
            degree = obj.Private_degree;
        end
        
        % Get kernel matrix
        function A_kern = get.A_kern(obj)
            % If the kernel matrix has already been computed, and no
            % parameters have changed, just return it.  Otherwise, we need
            % to recompute it
            if ~obj.recalc_kern_mtx
                A_kern = obj.A_kern;
            else
                % Compute kernel for all pairwise vector combinations in
                % A_feat
                A_feat = obj.Private_A_feat;
                M = size(A_feat, 1);
                
                A_kern = pdist(A_feat, obj.kernel_fxn_handle);
                A_kern = squareform(A_kern);    % Convert tril to symm mtx
                for m = 1:M
                    % Fill in missing diagonals
                    A_kern(m,m) = obj.kernel_fxn_handle(A_feat(m,:), ...
                        A_feat(m,:));
                end
                
                obj.A_kern = A_kern;
                obj.A_kern_sq = A_kern.^2;
                obj.recalc_kern_mtx = false;    % Clear recalc flag
            end
        end
        
        
        % *****************************************************************
        %                          LINTRANS METHODS
        % *****************************************************************
        
        % Size
        function [m,n] = size(obj)
            if nargout == 1
                m = size(obj.A_kern, 1);
                return
            else
                [m,n] = size(obj.A_kern);
            end
        end
        
        % Matrix multiply
        function z = mult(obj, x)
            z = obj.A_kern*x;
        end
        
        % Matrix multiply transpose
        function x = multTr(obj, y)
            x = obj.A_kern*y;       % Symmetric matrix; no transpose needed
        end
        
        % Matrix multiply with square
        function z = multSq(obj, x)
            z = obj.A_kern_sq*x;
        end
        
        % Matrix multiply transpose
        function x = multSqTr(obj, y)
            x = obj.A_kern_sq*y;    % Symmetric matrix; no transpose needed
        end
        
        
        % *****************************************************************
        %                          FEAT2KERN METHOD
        % *****************************************************************
        
        function B_kern = feat2kern(obj, B_feat)
            % Verify that input sizes make sense
            [M, N] = size(obj.Private_A_feat);
            [L, N2] = size(B_feat);
            if L == N  && N2 ~= N
                % Transpose input
                B_feat = B_feat.';
                L = N2;
            elseif N2 == N
                % We're okay, nothing to change
            else
                error('''B_feat'' must be an L-by-N feature matrix')
            end
            
            B_kern = NaN(L,M);
            for k = 1:L
                B_kern(k,:) = obj.kernel_fxn_handle(B_feat(k,:), ...
                    obj.Private_A_feat(:,:)).';
            end
        end
        
        
        % *****************************************************************
        %                      CROSSVAL PARTITION METHOD
        % *****************************************************************
        function [KLT_train, B_kern_test] = crossvalPartition(obj, ...
                partition, foldID)
            % Basic input sanity checking
            if nargin < 3
                error('crossvalPartition requires 2 arguments')
            else
                if isempty(partition)
                    fprintf('Creating a random partition w/ 10 folds\n')
                    partition = crossvalind('KFold', obj.size(), 10);
                elseif numel(partition) ~= obj.size();
                    error('Incorrect partition vector (dimension mismatch)')
                end
                
                if isempty(foldID)
                    error('Please supply a desired leave-out fold')
                end
            end
            
            % Get the feature matrix and partition into training and test
            % subsets
            A_train = obj.A_feat(partition ~= foldID, :);
            A_test = obj.A_feat(partition == foldID, :);
            kernel = obj.kernel;
            gamma = obj.gamma;
            offset = obj.offset;
            degree = obj.degree;
            
            % Create new training object
            KLT_train = KernelLinTrans(A_train, kernel, gamma, offset, degree);
            
            % Return kernel-space matrix of left-out test examples
            B_kern_test = KLT_train.feat2kern(A_test);
                    
        end
        
    end
        
end