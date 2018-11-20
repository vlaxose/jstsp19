classdef NeuralInEst < EstimIn
    % NeuralInEst:  Input estimation function for the neural decoding prob
    %
    % The input consists of two terms:  x = [z0; w(:)]
    %
    % where z0 = bias on the linear output modeled as a Gaussian
    % w = vector of weights modeled as a sparse
    
    properties
        % Estimator for the bias term x0, a AWGN estimator
        z0estim;
        
        % Estimator for the weights, x, a general DisScaEstim
        % with a Gauss-Bernoulli prior
        xestim;  % estimator for x
        
        % Dimensions
        ndly;   % number of delay
        nin;    % number of inputs
        nx;     % number of elements in linear weight
        
        % Nominal sparsity level
        rhoNom;
    end
    
    methods
        
        % Constructor
        %  z0mean mean of the bias term
        %  z0var  variance of the bias term
        %  xvar   variance of weights when non-zero
        %  rhoNom    probability that weights are non-zero
        %  nx     number of weights
        function obj = NeuralInEst(z0mean, z0var, xvar, rhoNom, ndly, nin)
            
            obj = obj@EstimIn;
            
            % Set dimensions
            obj.nin = nin;
            obj.ndly = ndly;
            obj.nx = nin*ndly;
            obj.rhoNom = rhoNom;
            
            % Construct estimator for weights, when they are "on"
            xmean = zeros(obj.nx,1);
            estim1 = AwgnEstimIn(xmean, xvar);
            
            % Apply group sparsity structure if rhoNom < 1.  Otherwise,
            % use the Gaussian values
            % grpInd is the group index 
            if (rhoNom < 1)
                rho = repmat(rhoNom, obj.nin, 1);   % sparsity level per group
                x0 = 0;                             % value when not on
                grpInd = repmat((1:nin),ndly,1);
                grpInd = grpInd(:);
                obj.xestim = GrpSparseEstim(estim1, rho, grpInd, x0);                        
            else
                obj.xestim = estim1;
            end
            
            % Construct estimator for bias z0
            obj.z0estim = AwgnEstimIn(z0mean, z0var);
            
        end
        
        % Get dimensions
        function [ndly,nin] = getDim(obj)
            nin = obj.nin;
            ndly = obj.ndly;
        end
        
        % Initial estimate based on the mean
        function [uhat, uvar, valInit] = estimInit(obj)
            
            % Initial estiamates for z0 and x
            [xhat0, xvar0, valInitx] = obj.xestim.estimInit();
            [z0hat0, z0var0, valInitz] = obj.z0estim.estimInit();
            
            % Pack results
            uhat = [xhat0; z0hat0];
            uvar = [xvar0; z0var0];
            valInit = valInitx + valInitz;
            
        end
                
        % AWGN estimation function
        % Provides the mean and variance of a variable u
        % from an observation v = u + w, w = N(0,wvar)
        function [umean, uvar, klNegDiv] = estim(obj, v, wvar)
            
            % Call estimator for x
            nx_ = obj.nx;
            [xmean,xvar, klNegDivx] = obj.xestim.estim(v(1:nx_), wvar(1:nx_) );
            
            % Call estiamtor for z0
            [z0mean, z0var, klNegDivz] = obj.z0estim.estim(v(nx_+1), wvar(nx_+1) );
            
            % Pack results
            umean = [xmean; z0mean];
            uvar = [xvar; z0var];
            klNegDiv = [klNegDivx; klNegDivz];
            
        end
        
        % Set sparsity level
        function setSparseProb(obj, rho)
            obj.xestim.setSparse(rho);
        end
    end
end

