classdef NeuralConnOut
    % True output function
    %   v = z - vbias +  N(0,zvar)
    %   rate = Sat( max(0, v-vbias) )
    %   w = scale*v;
    %   rate = w/(
    % wh
    
    properties
        % Bias and variance of input
        inputBias;
        inputVar;
        
        % Maximum rate
        applySat;   % boolean indicating if output is saturated or not
        rateMax;    % saturation level
        
        % Scale factor
        scale;
    end
    
    methods
        
        % Constructor
        % Sets parameters based on a sample of inputs x and the constraints
        %   tprob = P(x < inputBias).  tprob=0 implies no bias
        %   inSnr = medianPow / sqrt(inputVar)  [in linear scale]
        %   outSnr = scale*medianPow
        %   
        function obj = NeuralConnOut(x,tprob,inSnr,outSnr,outSnrMax)
            x = sort(x);
            nx = length(x);            
            if (tprob <= 0)
                obj.inputBias = 0;
            else
                obj.inputBias = x(round(tprob*nx));
            end
            medianPow = median(x) - obj.inputBias;
            obj.inputVar = sqrt( 10^(-0.1*inSnr)*medianPow );
            obj.scale = 10.^(0.1*outSnr)/ medianPow;
            if (nargin < 5)
                obj.applySat = false;
            else
                obj.applySat = ~isempty(outSnrMax);
            end
            if (obj.applySat)
                obj.rateMax = 10.^(0.1*outSnrMax);
            end
        end
        
        % Generate random cnt
        function [cnt, rate] = genRandCnt(obj, z)
            
            % Bias input
            nz = length(z);
            v = max(0, z - obj.inputBias + sqrt(obj.inputVar)*randn(nz,1));            
            
            % Saturate and scale rate      
            if (obj.applySat)            
                a = obj.scale/obj.rateMax;            
                rate = obj.rateMax*tanh(a*v);
            else
                rate = obj.scale*v;
            end
            
            % Generate count
            cnt = poissrnd( rate );
            
        end
        
        % Plot transfer function
        function rate = getTranserFn(obj, z)
            % Bias input            
            v = max(0, z - obj.inputBias);            
            
            % Saturate and scale rate
            if (obj.applySat)
                a = obj.scale/obj.rateMax;            
                rate = obj.rateMax*tanh(a*v);
            else
                rate = obj.scale*v;
            end
        end
    end
end