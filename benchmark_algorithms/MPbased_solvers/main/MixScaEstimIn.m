classdef MixScaEstimIn < handle
    % MixScaEstimIn:  Scalar estimator class constructed by mixing two other scalar estimators
    %
    % Let X1 and X0 be two random variables.
    % Then MixScaEstimIn is the estimator for a new random variable:
    %   X = { X1 with prob p1
    %       { X0 with prob 1-p1
    properties
        p1;                 % prob that estim = estim1
        estim1;             % Base estimator when U=1
        estim0;             % Base estimator when U=1
        autoTune = false;   % learning of parameters
        disableTune = false;% temporarily disable tuning of this and base estimators?
        tuneDim = 'joint';  % Determine dimension to autoTune over, must be 
                            % 'joint', 'col', or 'row'
        counter = 0;        % Counter to delay tuning
    end
    
    properties (Hidden)
        LogLikeFlag0 = false;   % estim0 implements loglikey method if true
        LogLikeFlag1 = false;   % estim1 implements loglikey method if true
        weightFlag0 = false;     %Indicates if estim0 has a weightFlag property
        weightFlag1 = false;     %Indicates if estim1 has a weightFlag property
    end
    
    
    methods
        % Constructor
        function obj = MixScaEstimIn(estim1, p1, estim0, varargin)
                obj.p1 = p1;
                obj.estim1 = estim1;
                obj.estim0 = estim0;
                for i = 1:2:length(varargin)
                    obj.(varargin{i}) = varargin{i+1};
                end
        end
        
        % Set method for estim0
        function set.estim0(obj, Estim0)
            % Check to ensure input Estim1 is a valid EstimIn class
            if isa(Estim0, 'EstimIn')
                if ismethod(Estim0, 'loglikey')
                    % Base estimator implements loglikey method
                    obj.LogLikeFlag0 = true;    %#ok<MCSUP>
                    obj.weightFlag0 = ~isempty(findprop(Estim0, 'mixWeight')); %#ok<MCSUP>
                end
                obj.estim0 = Estim0;
            else
                error('estim0 must be a valid EstimIn object')
            end
        end
        
        % Set method for estim1
        function set.estim1(obj, Estim1)
            % Check to ensure input Estim1 is a valid EstimIn class
            if isa(Estim1, 'EstimIn')
                if ismethod(Estim1, 'loglikey')
                    % Base estimator implements loglikey method
                    obj.LogLikeFlag1 = true;    %#ok<MCSUP>
                    obj.weightFlag1 = ~isempty(findprop(Estim1, 'mixWeight')); %#ok<MCSUP>
                end
                obj.estim1 = Estim1;
            else
                error('estim1 must be a valid EstimIn object')
            end
        end

        % Set method for disableTune
        function set.disableTune(obj, flag)
            if islogical(flag)
                % change property of this class 
                obj.disableTune = flag;
                % change property of each component class 
                if any(strcmp('disableTune', properties(obj.estim0)))
                    obj.estim0.disableTune = flag;
                end
                if any(strcmp('disableTune', properties(obj.estim1)))
                    obj.estim1.disableTune = flag;
                end
            else
                error('disableTune must be a boolean')
            end
        end

        % Compute prior mean and variance
        function [xhat, xvar, valInit] = estimInit(obj)
            [xhat1, xvar1, valInit1] = obj.estim1.estimInit;
            [xhat0, xvar0, valInit0] = obj.estim0.estimInit;
            xhat = obj.p1.*xhat1 + (1-obj.p1).*xhat0;
            xvar = obj.p1.*(xvar1 + abs(xhat1-xhat).^2) + ...
	    		(1-obj.p1).*(xvar0 + abs(xhat0-xhat).^2);
            valInit = obj.p1.*valInit1 + (1-obj.p1).*valInit0;	% check this!
        end
        
        % Compute posterior outputs
        function [xhat, xvar, klDivNeg, py1] = estim(obj, rhat, rvar)
            
            % Calculate posterior activity probabilities
            if ~obj.LogLikeFlag0
                % Convert from prob to log-prob domain
                loglike0 = log( obj.estim0.plikey(rhat, rvar) );
            else
                loglike0 = obj.estim0.loglikey(rhat, rvar);
            end
            
            if ~obj.LogLikeFlag1
                % Convert from prob to log-prob domain
                loglike1 = log( obj.estim1.plikey(rhat, rvar) );
            else
                loglike1 = obj.estim1.loglikey(rhat, rvar);
            end
            
            % Convert log-domain quantities into posterior activity
            % probabilities (i.e., py1 = Pr{X=X1 | y}, py0 = Pr{X=X0 | y})
            exparg = loglike0 - loglike1 + log(1 - obj.p1) - log(obj.p1);
            maxarg = 50; 
            exparg = max(min(exparg,maxarg),-maxarg); % numerical robustness
            py1 = (1 + exp(exparg)).^(-1);
            py0 = 1 - py1;
            
            %Update the mixture weight for estim1
            if obj.autoTune && ~obj.disableTune

              if (obj.counter>0), % don't tune yet
                obj.counter = obj.counter-1; % decrement counter 
              else % tune now

                [N, T] = size(rhat);
                %Average over all elements, per column, or per row
                switch obj.tuneDim
                    case 'joint'
                        obj.p1 = sum(py1(:))/N/T;
                    case 'col'
                        obj.p1 = repmat(sum(py1)/N,[N 1]);
                    case 'row'
                        obj.p1 = repmat(sum(py1,2)/T, [1 T]);
                    otherwise 
                        error('Invalid tuning dimension in SparseScaEstim');
                end
                %If subclass estim1 has mixWeights pass the posterior
                %"activity" of class 1
                if obj.weightFlag1
                    set(obj.estim1, 'mixWeight', py1./obj.p1)
                else
                    if ~isempty(findprop(obj.estim1,'autoTune'))
                      if obj.estim1.autoTune
                        warning(strcat('Auto tuning may fail: ',...
                                class(obj.estim1),...
                                ' does not include property mixWeight.'))
                      end 
                    end
                end
                %If subclass estim0 has mixWeights pass the posterior
                %"activity" of class 0
                if obj.weightFlag0
                    set(obj.estim0, 'mixWeight', py0./(1- obj.p1))
                else
                    if ~isempty(findprop(obj.estim0,'autoTune'))
                      if obj.estim1.autoTune
                        warning(strcat('Auto tuning may fail: ',...
                                class(obj.estim0),...
                                ' does not include property mixWeight.'))
                      end 
                    end
                end

              end
            end  
            
            % Compute posterior mean and variance
            [xhat1, xvar1, klDivNeg1] = obj.estim1.estim(rhat,rvar);
            [xhat0, xvar0, klDivNeg0] = obj.estim0.estim(rhat,rvar);
            xhat = py1.*xhat1 + py0.*xhat0;
            xvar = py1.*(abs(xhat1-xhat).^2 + xvar1) + ...
                py0.*(abs(xhat0-xhat).^2 + xvar0);
            
            % Compute negative K-L divergence
            if (nargout >= 3)
                klDivNeg = py1.*(klDivNeg1 + log(obj.p1./max(py1,1e-8))) ...
                    + py0.*(klDivNeg0 + log((1-obj.p1)./max(py0,1e-8)));
            end
            
        end
        
        % Generate random samples
        function x = genRand(obj, nx)
            x1 = obj.estim1.genRand(nx);
            x0 = obj.estim0.genRand(nx);
            p = rand(size(x1)) < obj.p1;
            x = x1.*p + x0.*(1-p);
        end
        
        % Get the points in the distribution
        function x0 = getPoints(obj)
            %x0 = [0; obj.estim1.getPoints()];				
            x0 = [obj.estim0.getPoints(); obj.estim1.getPoints()];	% not sure about this!
        end
        
        % Set sparsity level
        function setSparseProb(obj, p1)
            obj.p1 = p1;					% not sure if this needs to be modified!
        end
        
    end
    
end

