classdef MriLinTrans < LinTrans
    % Linear transformation for a MRI in the analysis mode
    %
    % This implements the function z = A(x) where A is the block matrix
    %
    %   A = [F; W; TV]
    %
    % where F = Fourier transform * coil sensitivities
    %       W = Wavelet
    %       TV = total variation
    %    
    properties (Access = private)
        % Parameter structure used by the Stanford code
        param;
        
        % Dimensions
        nrow, ncol;     % number of rows and columns
        
        % Fourier parameters
        ncoil;      % number of coils
        nsamp;      % number of rows after sub-sampling
                    % note that each col is fully sampled
                    
        mask;       % mask
        Isamp;      % indices of rows that are sampled
        noutF;      % number of outputs of the Fourier
        sens;       % coil sensitivity
        sensPow;    % coil sensitivity power
               
        % Wavelet and TV parts
        noutW;      % number of outputs of the Wavelet
        noutTV;     % number of outputs of the Total Variation
        unifVarTV = true;                    
    end
    
    methods 
        % Constructor based on the parameter structured created
        % by the Stanford code.
        function obj = MriLinTrans(param)
                       
            % Base class
            obj = obj@LinTrans;
            
            % Save parameter
            obj.param = param;
            
            % Get dimensions
            obj.nrow = size(param.y,1); 
            obj.ncol = size(param.y,2);
            obj.ncoil = size(param.y,3);
                    
            % Fourier part
            paramElement = getElement(param.E);
            obj.mask = paramElement{2};
            if (size(obj.mask,3) == 1)
                sampMask = obj.mask(:,1);            
                obj.Isamp = find(sampMask);
            else
                error('Coil dependent sampling not currently supported');
            end
            obj.nsamp = length(obj.Isamp);            
            obj.noutF = obj.nsamp * obj.ncoil * obj.ncol;

            % Save coil sensitivity matrix and power
            FourierElem = getElement(obj.param.E);
            obj.sens        = FourierElem{3};   
            obj.sensPow     = abs(obj.sens).^2;
            
            
            % Wavelet 
            obj.noutW = obj.nrow * obj.ncol;              
            
            % TV.  Note the multiplication by 2 for the row and col
            % difference
            obj.noutTV = 2 * obj.nrow * obj.ncol;    
                     
        end       
        
        % Set uniform variance flag for each components
        function setUnifVar(obj, val)
            obj.unifVarTV = val;
        end
        
        
        % Size
        function [nout,nin] = size(obj)
           nin = obj.nrow*obj.ncol;
           nout = obj.noutF + obj.noutW +  obj.noutTV;
        end
        
        % Extract the components sizes
        function [noutF, noutW, noutTV] = getCompSize(obj)
            noutF = obj.noutF;
            noutW = obj.noutW;
            noutTV = obj.noutTV;
        end
        
        % Divide an output vector into its components
        function [zF, zW, zTV] = extOutComp(obj, z)
            zF  = z(1:obj.noutF);
            zW  = z(obj.noutF+1:obj.noutF+obj.noutW);
            zTV = z(obj.noutF+obj.noutW+1:obj.noutF+obj.noutW+obj.noutTV);
        end
        
        % Pack components 
        function z = packOutComp(obj, zF, zW, zTV)
            z = zeros(obj.noutF+obj.noutW+obj.noutTV,1);
            z(1:obj.noutF) = zF;
            z(obj.noutF+1:obj.noutF+obj.noutW) = zW;
            z(obj.noutF+obj.noutW+1:obj.noutF+obj.noutW+obj.noutTV) = zTV;            
        end
        
        % Get mask
        function [Isamp] = getIsamp(obj)
            Isamp = obj.Isamp;
        end  

        % Matrix multiply:  z = A*x
        function [z] = mult(obj,x)
            
            % Reshape x to be square
            xsq = reshape(x, obj.nrow, obj.ncol);
            
            % Multiply using parmeter 
            zsqF = obj.param.E * xsq;
            
            % Subsample
            zsqF = zsqF(obj.Isamp,:,:);
            
            % Return to vector
            zF = zsqF(:);
            
            % Wavelet
            zsqW = obj.param.W*xsq;            
            zW = zsqW(:);
            
            % TV
            zsqTV = obj.param.TV*xsq;
            zTV = zsqTV(:);
            
            % Place into single column
            z = [zF; zW; zTV];
                  
          
        end
        
        % Multiply by square:  pvar = abs(Ad).^2*xvar
        function [pvar] = multSq(obj, xvar)

            % Reshape input
            xvarsq = reshape(xvar,obj.nrow,obj.ncol);
                        
            % Fourier part
            pvarsqZ = zeros(obj.nsamp,obj.ncol,obj.ncoil);
            for ch=1:obj.ncoil
                pvari = mean(mean( obj.sensPow(:,:,ch).*xvarsq ));
                pvarsqZ(:,:,ch) = pvari;
            end 
            pvarZ = pvarsqZ(:);            
            
            % Wavelet.  Use uniform variance approximation            
            pvarW = sum(xvarsq(:))/(obj.nrow*obj.ncol)*ones(obj.noutW,1);
            
            % TV 
            xvarrsh1 = xvarsq((2:end),:) + xvarsq((1:end-1),:);
            pvarsqTV1 = cat(1,xvarrsh1,zeros(1,obj.ncol));
            
            xvarrsh2 = xvarsq(:,(2:end)) + xvarsq(:,(1:end-1));
            pvarsqTV2 = cat(2,xvarrsh2,zeros(obj.nrow,1));
            
            pvarsqTV = cat(3, pvarsqTV1, pvarsqTV2);
            pvarTV = pvarsqTV(:);
            if (obj.unifVarTV)
                pvarTV = mean(pvarTV)*ones(obj.noutTV,1);
            end
            
            % Stack outputs
            pvar = [pvarZ;pvarW;pvarTV];
            
        end
        
        % Matrix multiply transpose:  x = A'*s
        function [x] = multTr(obj,s)
            
            % Extract components 
            [sF,sW,sTV] = obj.extOutComp(s);            
            
            % Fourier transform
            % -----------------
            % Reshape
            ssqZ = zeros(obj.nrow,obj.ncol,obj.ncoil);
            srshZ = reshape(sF, obj.nsamp, obj.ncol, obj.ncoil);
            ssqZ(obj.Isamp,:,:) = srshZ;
               
            % Perform transform manually
            % The operator param.E' does not implement a true
            % conjugate traspose.
            aux = zeros(obj.nrow,obj.ncol,obj.ncoil);
            if size(obj.mask,3)>1,  % Coil dependent sampling
                for ch=1:obj.ncoil,
                    aux(:,:,ch)=ifft2c_mri(ssqZ(:,:,ch).*obj.mask(:,:,ch));
                end
            else
                for ch=1:obj.ncoil,  % Coil independent sampling
                    aux(:,:,ch)=ifft2c_mri(ssqZ(:,:,ch).*obj.mask); 
                end
            end
            xsqZ=sum(aux.*conj(obj.sens),3);
            xsqZ(isnan(xsqZ))=0;   
            
            % Reshape
            xZ = xsqZ(:);
            
            % Wavelet transform
            % -------------------
            % Reshape
            ssqW = reshape(sW,obj.nrow,obj.ncol);            
                        
            % Perform transform
            xsqW = obj.param.W' * ssqW;
            
            % Reshape
            xW = xsqW(:);
            
            % TV transform
            % ------------
            ssqTV = reshape(sTV,obj.nrow,obj.ncol,2);

            % Perform transform
            xsqTV = obj.param.TV' * ssqTV;
            
            % Reshape
            xTV = xsqTV(:);
            
            % Sum results
            % -----------            
            x = xZ + xW + xTV;
        end
                    
 
        % Matrix multiply with componentwise square transpose:  
        %   rvar = (Ad.^2)'*svar
        function [rvar] = multSqTr(obj,svar)
            
            % Extract components
            [svarF, svarW, svarTV] = obj.extOutComp(svar);
           
            % Fourier part
            % ------------
            svarsqZ = zeros(obj.nrow,obj.ncol,obj.ncoil);
            svarrshZ = reshape(svarF, obj.nsamp, obj.ncol, obj.ncoil);
            svarsqZ(obj.Isamp,:,:) = svarrshZ;
            
            FourierElem = getElement(obj.param.E);
            FourierSenElem = FourierElem{3};
                     
            rvarsqZ = zeros(obj.nrow,obj.ncol);
            for ch=1:size(FourierSenElem,3)    
                rvarsqZ = rvarsqZ + obj.sensPow(:,:,ch)*sum(sum(svarsqZ(:,:,ch)))/(obj.ncol * obj.nrow);
            end 
            rvarZ = rvarsqZ(:);

            % Wavelet part
            svarsqW = reshape(svarW,obj.nrow,obj.ncol);
            rvarW = sum(svarsqW(:))/(obj.noutW)*ones((obj.nrow*obj.ncol),1);

            % Total variation part
            svarrshTV = reshape(svarTV,obj.nrow,obj.ncol,2);
            
            svarrshTV1 = svarrshTV(:,:,1);
            svarrshTV2 = svarrshTV(:,:,2);
            
            rvarrshTV1 = svarrshTV1(:,[1,1:end-1])+ svarrshTV1;
            rvarrshTV1(:,1) = svarrshTV1(:,1);
            rvarrshTV1(:,end) = svarrshTV1(:,1);
            
            rvarrshTV2 = svarrshTV2([1,1:end-1],:) + svarrshTV2;
            rvarrshTV2(1,:)= svarrshTV2(1,:);
            rvarrshTV2(end,:)= svarrshTV2(end-1,:);
            
            rvarrshTV = rvarrshTV1 + rvarrshTV2;
            rvarTV = rvarrshTV(:);
            
            % Sum parts            
            rvar = rvarZ + rvarW + rvarTV;
        end
                    
       
    end
    
end
