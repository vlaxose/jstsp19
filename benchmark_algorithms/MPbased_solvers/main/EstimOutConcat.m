classdef EstimOutConcat < EstimOut
    % EstimOutConcat:  Concatenation of output estimators
    %
    % The estimator creates an output estimator for a vector z = [z1 ... zn]
    % with estimators estimArray{i} 
    properties (Access = private)
        estimArray;  % Cell array of input estimators
        
        % Index array.  The components for zi are in z(ind(i):ind(i+1)-1)
        ind;         
        
        % number of columns
        ncol;
        hasLogLike; % hasLogLike(i) is true if we can call estimArray{i}.logLike(...)
        hasLogScale; % hasLogScale(i) is true if we can call estimArray{i}.logScale(...)
    end
     
    methods
        
        % Constructor
        % estimArray is a cell array of estimators
        % nz(i) is the number of elements for the estimator estimArray{i}.
        % If nz is not specified, then it will be determined by the size
        % operator applied to each element in estimArray.
        function obj = EstimOutConcat(estimArray, nz)    
            obj = obj@EstimOut;
            obj.estimArray = estimArray;
            
            % Check if nz was specified
            nzSpec = (nargin >= 2);
            
            % Record sizes 
            nelem = length(estimArray);
            obj.ind = zeros(nelem+1,1);
            obj.ind(1) = 1;
            for ielem = 1:nelem
                if (nzSpec)
                    %if ~ismethod(estimArray{ielem}, 'numColumns') || ~ismethod(estimArray{ielem}, 'logLike')
                    %    fprintf(2,'Note: EstimOutConcat arguments are not EstimOut objects. Adaptive step size will not work properly.\n');
                    %end

                    nzi = nz(ielem);
                    if ~ismethod(estimArray{ielem}, 'numColumns')
                        ncoli=1; % not really an EstimOut object
                    else
                        ncoli = estimArray{ielem}.numColumns();
                    end
                else
                    [nzi,ncoli] = size(estimArray{ielem});
                    if (isempty(nzi))                        
                        error(['Size of element ' int2str(ielem) ' not specified.  ' ...
                            'Call constructor with second argument to specify sizes, ' ...
                            'or make sure size operator is defined for the element'] );
                    end
                end
                obj.ind(ielem+1) = obj.ind(ielem)+nzi;
                if (ielem == 1)
                    obj.ncol = ncoli;
                elseif (ncoli ~= obj.ncol)
                    error(['Element ' int2str(ielem) ' has incorrect number '...
                        'of columns'] );
                end
            end
            for i = 1:nelem
                obj.hasLogLike(i) = ismethod(obj.estimArray{i}, 'logLike');
                obj.hasLogScale(i) = ismethod(obj.estimArray{i}, 'logScale');
            end
            
        end
                
        % Estimation function (see description in EstimOut.estim() )
        % Applied for each array element
        function [zhat,zvar] = estim(obj,phat,pvar)

            if isscalar(pvar)
                pvar = pvar * ones(size(phat));
            end
            
            % Initial estimates
            nelem = length(obj.estimArray);
            zhat = zeros(obj.ind(nelem+1)-1,obj.ncol);
            zvar = zeros(obj.ind(nelem+1)-1,obj.ncol);
                        
            for i = 1:nelem
                
                % Get estimates
                I = (obj.ind(i):obj.ind(i+1)-1)';
                [zhati, zvari] = obj.estimArray{i}.estim(phat(I,:),pvar(I,:));
                
                % Store results
                zhat(I,:) = zhati;
                zvar(I,:) = zvari;
            end
            
        end

        % Output cost used for adaptive stepsize
        function ll = logLike(obj,zhat,zvar) 
            if isscalar(zvar)
                zvar = zvar * ones(size(zhat));
            end
            
            % Initial estimates
            nelem = length(obj.estimArray);
            ll = [];
                        
            for i = 1:nelem
                
                % Get log likelihoods
                I = (obj.ind(i):obj.ind(i+1)-1)';
                if obj.hasLogLike(i)
                    lli =  obj.estimArray{i}.logLike(zhat(I,:), zvar(I,:));
                else
                    lli = 0; % not an EstimOut object -- note this will confound adaptive step sizes
                end
                % Concatenate results
		ll = [ll;lli];
            end
        end
       
        % Output cost used for adaptive stepsize
        function ll = logScale(obj,Axhat,pvar,phat)
            if isscalar(pvar)
                pvar = pvar * ones(size(Axhat));
            end
            
            % Initial estimates
            nelem = length(obj.estimArray);
            ll = [];
                        
            for i = 1:nelem
                
                % Get log scales
                I = (obj.ind(i):obj.ind(i+1)-1)';
                if obj.hasLogScale(i)
                    lli =  obj.estimArray{i}.logScale(Axhat(I,:), pvar(I,:), phat(I,:));
                else
                    lli = 0; % not an EstimOut object -- note this will confound adaptive step sizes
                end
                % Concatenate results
		ll = [ll;lli];
            end
        end
        
        % Size
        function [nz,ncol] = size(obj)
            nz = obj.ind(end)-1;
            ncol = obj.ncol;                
        end
       
        % Return number of columns 
        function S = numColumns(obj)
            S = obj.ncol;
        end

        % Generate random samples from each element p_{Y_i|Z_i}(y|z) in
        % EstimOutConcat
        function y = genRand(obj, z)

            % Retrieve number of elements
            nelem = length(obj.estimArray);
            %Preallocate space for z;
            y = zeros(size(z));

            %Call .genRand for each element in estimOutConcat.
            for i = 1:nelem
                % Get estimates
                I = (obj.ind(i):obj.ind(i+1)-1)';
                %Call individual .genRand
                y(I,:) =  obj.estimArray{i}.genRand(z(I,:));
            end

        end

 
    end
end

