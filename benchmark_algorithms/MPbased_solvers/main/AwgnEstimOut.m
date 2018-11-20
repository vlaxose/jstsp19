classdef AwgnEstimOut < EstimOut
    % AwgnEstimOut:  AWGN scalar output estimation function
    %
    % Corresponds to an output channel of the form
    %   y = scale*z + N(0, wvar)
    
    properties
        y;                  % measured output
        wvar;               % variance
        scale = 1;          % scale factor
        maxSumVal = false;  % True indicates to compute output for max-sum
        autoTune = false;   % Set to true for tuning of mean and/or variance
        disableTune = false;% Set to true to temporarily disable tuning
        tuneMethod = 'Bethe';  % Tuning method, in {ML,Bethe,EM0,EM}
        tuneDim = 'joint';  % Dimension to autoTune over, in {joint,col,row}
        tuneDamp = 0.1;     % Damping factor for autoTune in (0,1]
        counter = 0;        % Counter to delay tuning
        wvar_min = 1e-20;   % Minimum allowed value of wvar
    end
    
    methods
        % Constructor
        function obj = AwgnEstimOut(y, wvar, maxSumVal, varargin)
            obj = obj@EstimOut;
            if nargin ~= 0 % Allow nargin == 0 syntax
                obj.y = y;
                obj.wvar = wvar;
                if (nargin >= 3)
                    if (~isempty(maxSumVal))
                        obj.maxSumVal = maxSumVal;
                    end
                end
                if (nargin >= 4)
                    if isnumeric(varargin{1}) 
                        % make backwards compatible: 4th argument can specify the scale
                        obj.scale = varargin{1};
                    else
                        for i = 1:2:length(varargin)
                            obj.(varargin{i}) = varargin{i+1};
                        end
                    end 
                end
                
                %Warn user about zero-valued noise variance
                %if any(obj.wvar==0)
                %    warning(['Tiny non-zero variances will be used for'...
                %        ' computing log likelihoods. May cause problems'...
                %        ' with adaptive step size if used.']) %#ok<*WNTAG>
                %end
            end
        end
       
        % Set methods
        function obj = set.y(obj, y)
            assert(all(imag(y(:)) == 0), ...
                'AwgnEstimOut: y must be real valued.  Did you mean to use CAwgnEstimOut instead?');
                    % if we really want to handle real-valued noise and complex-valued y
                    % (and thus complex z), then we need to modify this file!
            obj.y = y;
        end

        function obj = set.wvar(obj, wvar)
            assert(all(wvar(:) >= 0), ...
                'AwgnEstimOut: wvar must be non-negative');
            obj.wvar = wvar;
        end

        function obj = set.wvar_min(obj, wvar_min)
            assert(all(wvar_min(:) > 0), ...
                'AwgnEstimOut: wvar_min must be positive');
            obj.wvar_min = wvar_min;
        end

        function obj = set.maxSumVal(obj, maxSumVal)
            assert(isscalar(maxSumVal)&&(ismember(maxSumVal,[0,1])||islogical(maxSumVal)), ...
                'AwgnEstimOut: maxSumVal must be a logical scalar');
            obj.maxSumVal = maxSumVal;
        end

        function obj = set.scale(obj, scale)
            assert(isnumeric(scale)&&isscalar(scale)&&(scale>0), ...
                'AwgnEstimOut: scale must be a positive scalar');
            obj.scale = scale;
        end

        function set.disableTune(obj, flag)
            assert(isscalar(flag)&&(ismember(flag,[0,1])||islogical(flag)), ...
                'AwgnEstimOut: disableTune must be a logical scalar');
            obj.disableTune = flag;
        end

        % Size
        function [nz,ncol] = size(obj)
            [nz,ncol] = size(obj.y);
        end
        
        % AWGN estimation function
        % Provides the posterior mean and variance of _real_ variable z
        % from an observation real(y) = scale*z + w
        % where z = N(phat,pvar) and w = N(0,wvar)
        function [zhat, zvar, partition] = estim(obj, phat, pvar)
           
            % Extract quantities
            y = obj.y;
            scale = obj.scale;
            phat_real = real(phat);
            scale2pvar = (scale^2)*pvar;

            % Compute posterior mean and variance
            wvar = obj.wvar;
            gain = pvar./(scale2pvar + wvar);
            zhat = (scale*gain).*(y-scale*phat_real) + phat_real;
            zvar = wvar.*gain;

            % Compute partition function
            if nargout==3
                partition = (1./sqrt(2*pi*(scale2pvar+wvar))).*exp(...
                                -(0.5*(phat_real-y).^2)./(scale2pvar+wvar) );
            end

            % Tune noise variance
            if obj.autoTune && ~obj.disableTune
                if (obj.counter>0), % don't tune yet
                    obj.counter = obj.counter-1; % decrement counter 
                else % tune now
                    [M,T] = size(phat);
                    damp = obj.tuneDamp;
                    %Learn variance, averaging over columns and/or rows
                    switch obj.tuneMethod 

                      case 'ML' % argmax_wvar Z(y;wvar)=\int p(y|z;wvar) N(z;phat,pvar) dz
                        switch obj.tuneDim
                          case 'joint'
                            wvar1 = mean((scale*phat_real(:)-y(:)).^2 - scale2pvar(:));
                            wvar0 = max(obj.wvar_min, wvar1)*ones(size(obj.wvar));
                          case 'col'
                            wvar1 = (1/M)*sum((scale*phat_real-y).^2 - scale2pvar,1);
                            wvar0 = ones(M,1)*max(obj.wvar_min, wvar1);
                          case 'row'
                            wvar1 = (1/T)*sum((scale*phat_real-y).^2 - scale2pvar,2);
                            wvar0 = max(obj.wvar_min, wvar1)*ones(1,T);
                          otherwise
                            error('Invalid tuning dimension in AwgnEstimOut');
                        end
                        if damp==1
                          obj.wvar = wvar0;
                        else % apply damping
                          obj.wvar = exp( (1-damp)*log(obj.wvar) + damp*log(wvar0));
                        end

                      case 'Bethe' % Method from Krzakala et al J.Stat.Mech. 2012
                        svar = 1./(scale2pvar + obj.wvar);
                        shat = (y-scale*phat_real).*svar;
                        switch obj.tuneDim
                          case 'joint'
                            ratio = sum(shat(:).^2)/sum(svar(:));
                            if damp~=1, ratio = ratio.^damp; end;
                            obj.wvar = max(obj.wvar_min, obj.wvar*ratio);
                          case 'col'
                            ratio = sum(shat.^2,1)./sum(svar,1);
                            if damp~=1, ratio = ratio.^damp; end;
                            obj.wvar = max(obj.wvar_min, obj.wvar.*(ones(M,1)*ratio));
                          case 'row'
                            ratio = sum(shat.^2,2)./sum(svar,2);
                            if damp~=1, ratio = ratio.^damp; end;
                            obj.wvar = max(obj.wvar_min, obj.wvar.*(ratio*ones(1,T)));
                          otherwise
                            error('Invalid tuning dimension in AwgnEstimOut');
                        end

                      case 'EM0'
                        switch obj.tuneDim
                          case 'joint'
                            wvar1 = mean((y(:)-zhat(:)).^2);
                            wvar0 = max(obj.wvar_min, wvar1)*ones(size(obj.wvar));
                          case 'col'
                            wvar1 = (1/M)*sum((y-zhat).^2,1);
                            wvar0 = ones(M,1)*max(obj.wvar_min, wvar1);
                          case 'row'
                            wvar1 = (1/T)*sum((y-zhat).^2,2);
                            wvar0 = max(obj.wvar_min, wvar1)*ones(1,T);
                          otherwise
                            error('Invalid tuning dimension in AwgnEstimOut');
                        end 
                        if damp==1
                          obj.wvar = wvar0;
                        else % apply damping
                          obj.wvar = exp( (1-damp)*log(obj.wvar) + damp*log(wvar0));
                        end

                      case 'EM'
                        switch obj.tuneDim
                          case 'joint'
                            wvar1 = mean((y(:)-zhat(:)).^2 + zvar(:));
                            wvar0 = max(obj.wvar_min, wvar1)*ones(size(obj.wvar));
                          case 'col'
                            wvar1 = (1/M)*sum((y-zhat).^2 + zvar,1);
                            wvar0 = ones(M,1)*max(obj.wvar_min, wvar1);
                          case 'row'
                            wvar1 = (1/T)*sum((y-zhat).^2 + zvar,2);
                            wvar0 = max(obj.wvar_min, wvar1)*ones(1,T);
                          otherwise
                            error('Invalid tuning dimension in AwgnEstimOut');
                        end 
                        if damp==1
                          obj.wvar = wvar0;
                        else % apply damping
                          obj.wvar = exp( (1-damp)*log(obj.wvar) + damp*log(wvar0));
                        end

                      otherwise
                        error('Invalid tuning method in AwgnEstimOut');
                    end
                end
            end

        end
        
        % Compute output cost:
        % For sum-product compute
        %   E_Z( log p_{Y|Z}(y|z) ) with Z ~ N(phat, pvar)
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = phat
        function ll = logLike(obj,phat,pvar)
            
            % Ensure variance is small positive number
            wvar_pos = max(obj.wvar_min, obj.wvar);
            
            % Get scale
            scale = obj.scale;
            
            % Compute log-likelihood
            if ~(obj.maxSumVal)
                predErr = ((obj.y-scale*real(phat)).^2 + (scale^2)*pvar)./wvar_pos;
            else
                predErr = ((obj.y-scale*real(phat)).^2)./wvar_pos;
            end
            ll = -0.5*(predErr); %return the values without summing
        end
        
        % Compute output cost:
        %   (Axhat-phatfix)^2/(2*pvar*alpha) + log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
        %   with phatfix such that Axhat=alpha*estim(phatfix,pvar) + (1-alpha)*phatfix.
        % For max-sum GAMP, compute
        %   log p_{Y|Z}(y|z) @ z = Axhat
        function ll = logScale(obj,Axhat,pvar,phat,alpha)  %#ok<INUSL>
            
            %Set alpha if not provided to unity
            if nargin < 5
                alpha = 1;
            end
            
            % Ensure variance is small positive number
            wvar1 = max(obj.wvar_min, obj.wvar);
            
            %Get the scale
            s = obj.scale;
            
            % Compute output cost
            if ~(obj.maxSumVal)
                
                %Use closed form
                closed_form = true;
                
                % Compute output cost
                if any(alpha ~= 1)
                    alphaFlag = true;
                else
                    alphaFlag = false;
                end
                if closed_form
                    %ptil = (pvar./wvar1+1).*Axhat - (pvar./wvar1).*obj.y;
                    %Old closed form update without scale
                    %                     ll = -0.5*( log(pvar+wvar1) + (obj.y-real(Axhat)).^2./wvar1 ...
                    %                         + log(2*pi) );
                    
                    % Compute in closed form
                    if ~alphaFlag
                        ll = -0.5*( log(s^2*pvar + wvar1) ...
                            + (obj.y - s*real(Axhat)).^2./wvar1 + log(2*pi));
                    else
                        ll = -0.5*( log(s^2*pvar + wvar1) ...
                            + (obj.y - s*real(Axhat)).^2./(wvar1 + pvar.*(1-alpha)) + log(2*pi));
                    end
                else
                    % Find the fixed-point of phat
                    opt.phat0 = Axhat; % works better than phat
                    opt.alg = 1; % approximate newton's method
                    opt.maxIter = 3;
                    opt.tol = 1e-4;
                    opt.stepsize = 1;
                    opt.regularization = obj.wvar^2;  % works well up to SNR=160dB
                    opt.debug = false;
                    opt.alpha = alpha;
                    [phatfix,zhat] = estimInvert(obj,Axhat,pvar,opt);
                    
                    % Compute log int_z p_{Y|Z}(y|z) N(z;phatfix, pvar)
                    ls = -0.5*log(2*pi*(obj.wvar + s^2*pvar)) ...
                        - (obj.y - s*real(phatfix)).^2 ./ (2*(obj.wvar + s^2*pvar));
                    
                    % Combine to form output cost
                    ll = ls + (0.5*(real(zhat - phatfix)).^2./pvar).*alpha;

                end
                
            else
                % Output cost is simply the log likelihood
                ll = -0.5*((obj.y-s*real(Axhat)).^2)./wvar1;
            end
        end
        
        function S = numColumns(obj)
            %Return number of columns of Y
            S = size(obj.y,2);
        end
        
        % Generate random samples from p(y|z)
        function y = genRand(obj, z)
            y = sqrt(obj.wvar).*randn(size(z)) + obj.scale.*z;
        end
    end
    
end

