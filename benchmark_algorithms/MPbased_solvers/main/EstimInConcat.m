classdef EstimInConcat < EstimIn
    % EstimInConcat:  Concatenation of input estimators
    %
    % The estimator creates an input estimator for a vector x = [x1 ... xn]
    % with estimators estimArray{i} 
    properties
        estimArray;  % Cell array of input estimators
        
        % Index array.  The components for xi are in x(ind(i):ind(i+1)-1)
        ind;         
    end
    
    methods
        
        % Constructor
        % estimArray is a cell array of estimators
        % nx(i) is the number of elements for the estimator estimArray{i}.
        function obj = EstimInConcat(estimArray, nx)    
            obj = obj@EstimIn;
            obj.estimArray = estimArray;
            
            % Constructor 
            nelem = length(estimArray);
            obj.ind = zeros(nelem+1,1);
            obj.ind(1) = 1;
            for ielem = 1:nelem
                obj.ind(ielem+1) = obj.ind(ielem)+nx(ielem);
            end
            
        end
                
        % Compute prior mean and variance
        function [xhat, xvar, valInit] = estimInit(obj)
            
            xhat1 = obj.estimArray{1}.estimInit();
            num_col = size(xhat1,2);
            nelem = length(obj.estimArray);
	        num_row = obj.ind(nelem+1)-1;
              
            xhat = zeros(num_row,num_col);
            xvar = zeros(num_row,num_col);
            valInit = 0;
                        
            for i = 1:nelem
                % Get estimates
                I = (obj.ind(i):obj.ind(i+1)-1)';
                [xhati, xvari, valIniti] = obj.estimArray{i}.estimInit();
                
                % Store results
                xhat(I,:) = xhati;
                xvar(I,:) = xvari;
                valInit = valInit + sum(valIniti(:));                
            end
            
        end

        % Estimator 
        function [xhat, xvar, val] = estim(obj, rhat, rvar)
            
            nelem = length(obj.estimArray);
            xhat = zeros(size(rhat));
            xvar = zeros(size(rhat));
            val = zeros(size(rhat));
            
            for i = 1:nelem
                % Get estimates
                I = (obj.ind(i):obj.ind(i+1)-1)';
                switch numel(rvar)
                    
                    case numel(rhat) %all values provided
                        [xhati, xvari, vali] = ...
                            obj.estimArray{i}.estim(rhat(I,:), rvar(I,:));
                        
                    case nelem %1 for each distribution
                        [xhati, xvari, vali] = ...
                            obj.estimArray{i}.estim(rhat(I,:), rvar(i));
                        
                    case 1 %single value, use it for all
                        [xhati, xvari, vali] = ...
                            obj.estimArray{i}.estim(rhat(I,:), rvar);
                        
                    otherwise
                        error('Cannot interpret provided variance dimensions')
                end
                
                % Store results
                xhat(I,:) = xhati;
                xvar(I,:) = xvari;
                val(I,:) = vali;                
            end
        end
       
        % Generate random samples from the distribution
        function x = genRand(obj, outSize)

            nelem = length(obj.estimArray);
	        num_row = obj.ind(nelem+1)-1;

            if outSize ~= num_row
                error('Number of random samples must match number of components in concatenated EstimIn.')
            end

            x1 = obj.estimArray{1}.genRand(1);
	        num_col = size(x1,2);
	        x = zeros(num_row,num_col);

	        for i = 1:nelem
	            % Get random samples
                I = (obj.ind(i):obj.ind(i+1)-1)';
                xi = obj.estimArray{i}.genRand(length(I));

                % Store results 
                x(I,:) = xi;
	        end
        end

    end
end

