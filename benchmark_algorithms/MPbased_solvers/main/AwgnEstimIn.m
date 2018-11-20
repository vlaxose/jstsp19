classdef AwgnEstimIn < EstimIn
    % AwgnEstimIn:  AWGN scalar input estimation function
    
    properties 
        var0_min = eps;     % Minimum allowed value of var0
        mean0 = 0;          % Prior mean 
        var0 = 1;           % Prior variance 
        maxSumVal = false;  % True indicates to compute output for max-sum
        autoTune = false;   % Set to true for taut tuning of params
        disableTune = false;% Set to true to temporarily disable tuning
        mean0Tune = true;   % Enable Tuning of mean0
        var0Tune = true;    % Enable Tuning of var0
        tuneDim = 'joint';  % Determine dimension to autoTune over, in {joint,col,row}
        counter = 0;        % Counter to delay tuning
    end
    
    properties (Hidden)
        mixWeight = 1;              % Weights for autoTuning
    end
    
    methods
        % Constructor
        function obj = AwgnEstimIn(mean0, var0, maxSumVal, varargin)
            obj = obj@EstimIn;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.mean0 = mean0;
                obj.var0 = var0;
                if (nargin >= 3)
                    if (~isempty(maxSumVal))
                        obj.maxSumVal = maxSumVal;
                    end
                end
                for i = 1:2:length(varargin)
                    obj.(varargin{i}) = varargin{i+1};
                end
                
                % warn user about inputs
                %if any(~isreal(mean0(:))),
                %    error('First argument of AwgnEstimIn must be real-valued');
                %end;
                %if any((var0(:)<0))||any(~isreal(var0(:))),
                %    error('Second argument of AwgnEstimIn must be non-negative');
                %end;
            end
        end
        
        %Set Methods
        function obj = set.var0_min(obj, var0_min)
            assert(all(var0_min(:) > 0), ...
                'AwgnEstimIn: var0_min must be positive');
            obj.var0_min = var0_min; 
        end

        function obj = set.mean0(obj, mean0)
            assert(all(isreal(mean0(:))), ...
                'AwgnEstimIn: mean0 must be real-valued');
            obj.mean0 = mean0;
        end
        
        function obj = set.var0(obj, var0)
            assert(all(var0(:) > 0), ...
                'AwgnEstimIn: var0 must be positive');
            obj.var0 = max(obj.var0_min,var0); % avoid too-small variances!
        end
        
        function obj = set.mixWeight(obj, mixWeight)
            assert(all(mixWeight(:) >= 0), ...
                'AwgnEstimIn: mixWeights must be non-negative');
            obj.mixWeight = mixWeight;
        end
        
        function obj = set.maxSumVal(obj, maxsumval)
            assert(isscalar(maxsumval)&&(ismember(maxsumval,[0,1])||islogical(maxsumval)), ...
                'AwgnEstimIn: maxSumVal must be a logical scalar');
            obj.maxSumVal = maxsumval;
        end

        function set.disableTune(obj, flag)
            assert(isscalar(flag)&&(ismember(flag,[0,1])||islogical(flag)), ...
                'AwgnEstimIn: disableTune must be a logical scalar');
            obj.disableTune = flag;
        end

        % Prior mean and variance
        function [mean0, var0, valInit] = estimInit(obj)
            mean0 = obj.mean0;
            var0  = obj.var0;
            valInit = 0;
        end

        % Size
        function [nx,ncol] = size(obj)
            [nx,ncol] = size(obj.mean0);
        end

        % AWGN estimation function
        % Provides the mean and variance of a variable x = N(uhat0,uvar0)
        % from an observation real(rhat) = x + w, w = N(0,rvar)
        function [xhat, xvar, val] = estim(obj, rhat, rvar)
            % Get prior
            uhat0 = obj.mean0;
            uvar0 = obj.var0; 
            
            % Compute posterior mean and variance
            gain = uvar0./(uvar0+rvar);
            xhat = gain.*(real(rhat)-uhat0)+uhat0;
            xvar = gain.*rvar;
            
            if obj.autoTune && ~obj.disableTune

              if (obj.counter>0), % don't tune yet
                obj.counter = obj.counter-1; % decrement counter 
              else % tune now

                [N, T] = size(rhat);
                %Learn mean if enabled 
                if obj.mean0Tune
                    %Average over all elements, per column, or per row
                    switch obj.tuneDim
                        case 'joint'
                            obj.mean0 = sum(obj.mixWeight(:).*xhat(:))/N/T;
                        case 'col'
                            obj.mean0 = repmat(sum(obj.mixWeight.*xhat)/N, [N 1]);
                        case 'row'
                            obj.mean0 = repmat(sum(obj.mixWeight.*xhat,2)/T, [1 T]);
                        otherwise 
                            error('Invalid tuning dimension in AwgnEstimIn');
                    end
                end
                %Learn variance if enabled 
                if obj.var0Tune
                    %Average over all elements, per column, or per row
                    switch obj.tuneDim
                        case 'joint'
                            obj.var0 = sum(obj.mixWeight(:)...
                                .*(xhat(:) - obj.mean0(:)).^2 + xvar(:))/(N*T);
                        case 'col'
                            obj.var0 = repmat(sum(obj.mixWeight...
                                .*(xhat - obj.mean0).^2 + xvar, 1)/N, [N 1]);
                        case 'row'
                            obj.var0 = repmat(sum(obj.mixWeight...
                                .*(xhat - obj.mean0).^2 + xvar, 2)/T, [1 T]);
                        otherwise 
                            error('Invalid tuning dimension in AwgnEstimIn');
                    end
                    %uvar0 = max(obj.var0_min,obj.var0);
                end

              end
            end  
            
            if (nargout >= 3)                            
                if ~(obj.maxSumVal)
                    % Compute the negative KL divergence            
                    %   klDivNeg = \sum_i \int p(x|r)*\log( p(x) / p(x|r) )dx
                    xvar_over_uvar0 = rvar./(uvar0+rvar);
                    val = 0.5*(log(xvar_over_uvar0) + (1-xvar_over_uvar0) ...
                        - (xhat-uhat0).^2./uvar0 );
                else
                    % Evaluate the (log) prior
                    val = -0.5* (xhat-uhat0).^2./uvar0;
                end
            end

        end
        
        % Generate random samples
        function x = genRand(obj, outSize)
            if isscalar(outSize)
                x = sqrt(obj.var0).*randn(outSize,1) + obj.mean0;
            else
                x = sqrt(obj.var0).*randn(outSize) + obj.mean0;
            end
        end
        
        % Computes the likelihood p(rhat) for real(rhat) = x + v, v = N(0,rvar)
        function py = plikey(obj,rhat,rvar)
            py = exp(-1./(2*(obj.var0+rvar)).*(real(rhat)-obj.mean0).^2);
            py = py./ sqrt(2*pi*(obj.var0+rvar));
        end
        
        % Computes the log-likelihood, log p(rhat), for real(rhat) = x + v, where 
        % x = N(obj.mean0, obj.var0) and v = N(0,rvar)
        function logpy = loglikey(obj, rhat, rvar)
            logpy = (-0.5)*( log(2*pi) + log(obj.var0 + rvar) + ...
                ((real(rhat) - obj.mean0).^2) ./ (obj.var0 + rvar) );
        end

    end
    
end

