classdef GrpSparseEstim  < EstimIn
    % GrpSparseEstim:  Input estimator for vectors with group sparsity
    %
    % The estimator is based on the random vector X generated as follows:  
    % Each component X(j) is associated with a group index, grpInd(j),
    % where grpInd(j) = 1,...,ngrp.  The components of the random vector X
    % are then generated as:
    %
    %   X(j) = X1(j) when U(k)=1 
    %        = x0(j) when U(k)=0
    %
    % where k = grpInd(j) is the group index; U(k) is a binary variable
    % indicating whether the group k is "on" or "off";  X1(j) are a set of
    % independent random variables describing the distribution when the 
    % group is on; and x0(j) are a set of constants for the values of X(j)
    % when the group is off.  It is assumed that U(k) are iid with
    % P(U(k)=1) = p1(k).    
    properties                
        % Base estimator when U = 1
        estim1;
        
        % Sparse scalar estimator for X
        sparseEstim;  
        x0;
                
        % Group indices 
        grpInd;
        
        % Nominal sparsity level
        p1Nom;
    end
    
    methods
        
        % Constructor
        %  estim1  Estimator for the the "on" variables, X1 above.
        %  x0      Values when the variables are "off"
        %  p1Nom   Nominal values for the probability of being "on"
        %  grpInd  Group indices (see above).
        function obj = GrpSparseEstim(estim1, p1Nom, grpInd, x0)
            
            % Get defaults
            if (nargin < 4)
                x0 = 0;
            end
            
            % Construct the base estimation class
            obj = obj@EstimIn;
                        
            % Construct estimator for weights     
            p1Nomx = p1Nom(grpInd);
            obj.sparseEstim = SparseScaEstim( estim1, p1Nomx, x0 );
            
            % Save other parameters
            obj.estim1 = estim1;
            obj.p1Nom = p1Nom;
            obj.grpInd = grpInd;
            obj.x0 = x0;
            
        end
        
        % Get dimensions
        function [xhat, xvar, valInit] = estimInit(obj)
            
            % Set the sparsity probability to the initial nominal value
            obj.sparseEstim.setSparseProb( obj.p1Nom(obj.grpInd) );
            
            % Get initial values
            [xhat, xvar, valInit] = obj.sparseEstim.estimInit();
            
        end               
        
        % AWGN estimation function.
        % Provides the mean and variance of X given observations of 
        % the form R = X + V, where V = N(0,rvar) and R=rhat.    
        function [xmean, xvar, val] = estim(obj, rhat, rvar)
            
            % Update the sparsity levels
            % --------------------------
            
            % Compute likelihoods:
            % pr1 = P(R=rhat|U=1) and pr0 = P(R=rhat|U=0)
            pr1 = obj.estim1.plikey(rhat,rvar);  
            pr0 = exp(-(rhat-obj.x0).^2./(2*rvar))./sqrt(2*pi*rvar);
            
            % Compute LLR message from the variable nodes
            logLikeIn = log(pr1./max(pr0, 1e-8) );
            
            % Compute the LLR messages back to the variable nodes
            nx   = length(logLikeIn);
            ngrp = max(obj.grpInd);
            logLikeOut = zeros(nx,1);      
            logLikeSum = zeros(ngrp,1);
            ngrp = max(obj.grpInd);
            for igrp = 1:ngrp
                I = (obj.grpInd == igrp);
                logLikeSum(igrp) = sum( logLikeIn.*I) + log(obj.p1Nom(igrp));
                logLikeOut = logLikeOut.*(1-I) + logLikeSum(igrp).*I;
            end
            logLikeOut = logLikeOut - logLikeIn;
            
            % Set the new sparse probability
            rho = 1./(1+exp(-logLikeOut));
            rho = min(max( rho, 1e-3), 1-1e-3);
            rho = rho(:);
                        
            % Compute the total probability
            rhoTot = 1./(1+exp(-logLikeSum));
            rhoTot = min(max( rhoTot, 1e-3), 1-1e-3);
            
            % Update the probabilities
            obj.sparseEstim.setSparseProb(rho);

            % Call estimator based on the updated sparsity probabilities
            % -----------------------------------------------------------
            [xmean,xvar,val,py1] = obj.sparseEstim.estim(rhat, rvar);
            
            % Compute the K-L divergence due to the sparsity
            klDiv = py1.*log(rho./max(py1,1e-8)) + ...
                (1-py1).*log((1-rho)./max(1-py1,1e-8));
            
            % Subtract the entropy computed per component
            val = val - klDiv;
            
            % Add the entropy per group
            val(1) = val(1) + sum( rhoTot.*log(obj.p1Nom./max(rhoTot,1e-8)) ...
                + (1-rhoTot).*log((1-obj.p1Nom)./max(1-rhoTot,1e-8)) );
            
        end
        
        % Generate a random vector with the group sparsity
        function x = genRand(obj, nx)
            
            % Generate random samples for the case when U = 1
            x = obj.estim1.genRand(nx);
            
            % Generate sparse support
            ngrp = max(obj.grpInd);
            u = (rand(ngrp,1) < obj.p1Nom);
            u = u(obj.grpInd((1:nx)'));
            
            % Apply sparsity
            x = x.*u + obj.x0.*(1-u);
            
        end
        
    end
    
    
    
end