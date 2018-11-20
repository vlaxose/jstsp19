classdef NeuralParam < handle
    
    properties
        linWt;      % Linear weight
        p;          % Polynomial coefficients in NL model
        noiseVar;   % noise variance after linear summing
    end
    
    methods
        % Constructor
        function obj = NeuralParam(linWt, p, noiseVar)
            obj.p = p;
            obj.linWt = linWt;
            obj.noiseVar = noiseVar;
        end
        
        % Copy
        function obj1 = copy(obj)
            obj1 = NeuralParam(obj.linWt, obj.p, obj.noiseVar);
        end
        
        % Plot non-linear component
        function [z,rate] = plotNL(obj)
            
            % Get mean and variance of z under iid Bernouli excitation
            probAct = 0.5;
            zmean = probAct*sum(sum(obj.linWt));
            zvar = probAct*(1-probAct)*sum(sum(obj.linWt.^2)) + obj.noiseVar;
            
            % Get z points to plot
            z = linspace(-4,4,1000)'*sqrt(zvar) + zmean;
            
            % Plot rate
            v = polyval(obj.p,z);
            rate = log(1+exp(v));
            plot(z, rate);            
            grid on;
        end
        
        % Plot linear weight
        function plotLinWt(obj)
            plot(obj.p(1)*obj.linWt);
        end
    end
end