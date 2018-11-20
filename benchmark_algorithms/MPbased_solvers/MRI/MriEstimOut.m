classdef MriEstimOut < EstimOutConcat
        
    methods
        % Constructor
        function obj = MriEstimOut(param, autoScale)
            
            % Parameters
            if (nargin < 2)
                autoScale = false;
            end

            % Get dimensions
            nrow = size(param.y,1); 
            ncol = size(param.y,2);
            ncoil = size(param.y,3);

            % Construct Fourier Estimator based on sampled data
            paramElement = getElement(param.E);
            sampMask = paramElement{2}(:,1);
            Isamp = find(sampMask);
            ysamp = param.y(Isamp,:,:);
            nsamp = length(Isamp);                        
            noutF = nsamp * ncol * ncoil;            
            vary = ones(noutF,1);           
            estF = CAwgnEstimOut(ysamp(:),vary,true);
            
            % Use L1-estimators for wavelet and TV parts
            estW = L1EstimOut(param.L1Weight);            
            estTV = L1EstimOut(param.TVWeight);
            if (autoScale)
                estW.setAutoScale( param.L1Weight*0.1, param.L1Weight*10);
                estTV.setAutoScale( param.TVWeight*0.1, param.TVWeight*10);
            end            
            % Get dimensions for wavelet and TV parts
            % Note multiplication by 2 for the TV part since for the row
            % and column differences
            noutW = nrow * ncol;
            noutTV = 2 * nrow * ncol;
            
            % Construct array
            estimArray = {estF, estW, estTV};
            nz = [noutF noutW noutTV]';            
            obj = obj@EstimOutConcat(estimArray, nz);
            
        end
        
    end
    
end
