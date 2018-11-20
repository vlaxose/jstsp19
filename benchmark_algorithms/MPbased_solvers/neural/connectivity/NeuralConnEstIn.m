classdef NeuralConnEstIn < EstimInConcat
    % NeuralConnEstIn:  Input estimator for the connectivity problem
    
    methods
        % Constructor
        %   rho:  sparsity level for the connectivity
        %   nwt: number of weights
        %   estimWt1:  estimate of weights when not sparse
        %   z0mean, z0var:  mean and variance of bias
        function obj = NeuralConnEstIn(nwt,estimWt,z0mean, z0var)
            
            % Estimator for bias
            estimZ0 = AwgnEstimIn(z0mean, z0var);
            
            % Create array
            estimArray = cell(2,1);
            estimArray{1} = estimWt;
            estimArray{2} = estimZ0;
            nx = [nwt 1]';
            
            % Create concatanated estiamtor
            obj = obj@EstimInConcat(estimArray, nx);
            
        end
    end
end