classdef LinTransWavelet < LinTrans
    % This transforms an image (flattened into a 1d vector)
    % via one or more 2d Daubechies wavelet bases
    % 
    properties
        S; % cell array of size info
        numBases; % number of dbX wavelet bases
        nlevel; % number of wavelet levels
        wname;  % wavelet names
        ncoef; % array of sizes for each wavelet base
        skips; % number of initial coefficients to skip from each base
        opts;  %valid options: 
               %     skipLevels=1  omits the wavelet "Approximation" coefficients
               %     skipLevels=2  also omit the first level "Detail" coefficients
               %     scale=N  scales the linear operator by N
    end

    methods
        function obj = LinTransWavelet( dims, whichBases,nlevel,opts )
            if nargin < 4
                opts=struct();
            end
            if ~isfield(opts,'skipLevels')
                opts.skipLevels = 0;
            end
            if ~isfield(opts,'scale')
                opts.scale = 1.0;
            end

            if length(dims)==1
                dims=[dims 1];
            end
            
            %Count the number of requested bases
            numBases = length(whichBases);
            
            im = rand(dims);
            
            C=cell(numBases,1);
            S=cell(numBases,1);
            wname=cell(numBases,1);
            ncoef=zeros(numBases,1);

            dwtmode('per');
            skips=zeros(numBases,1);

            for k=1:numBases
                wname{k} = sprintf('db%d',whichBases(k));
                [C{k},S{k}] = wavedec2(im,nlevel,wname{k});
                ncoef(k) = numel(C{k});
                for i=1:nlevel
                    nDetCo = prod( S{k}(i+1,:) );
                end
                if opts.skipLevels>0
                    skips(k) =  prod( S{k}(1,:) ); % Approximation coefficients
                    for i=2:opts.skipLevels
                        skips(k) = skips(k) +  prod( S{k}(i,:) );
                    end
                end
            end

            numCo = sum(ncoef) - sum(skips);
            obj = obj@LinTrans( numCo, prod(dims) , sqrt(numCo)*opts.scale);

            obj.S = S;
            obj.numBases = numBases;
            obj.nlevel = nlevel;
            obj.wname = wname;
            obj.ncoef = ncoef;
            obj.opts = opts;
            obj.skips = skips;
        end

        function CoefGroupIx = GetGroupIndices(obj) 
            % CoefGroupIx{1} is the array of the Approximation coefficients
            % CoefGroupIx{k+1} is the array of the detail coefficients for the db k dictionary
            nlevel = obj.nlevel;
            Aix = cell(obj.numBases,1);  % indices of the wavelet Approximation coefficients
            Dix = cell(obj.numBases,nlevel); % indices of the wavelet Detail coefficients at each level
            ct=0;
            for k=1:obj.numBases
                nAppCo = prod( obj.S{k}(1,:) ); % the number of approximation coefficients at this level
                Aix{k} = (1:nAppCo)' + ct;  % the indices of the detail coefficients for the db k dictionary
                ct = ct + nAppCo;
                for i=1:nlevel
                    nDetCo = prod( obj.S{k}(i+1,:) );
                    Dix{k,i} = (1:3*nDetCo)' + ct; % also possible to split up into Horiz,Vert,Diag detail coeffs
                    ct = ct + 3 * nDetCo;
                end
            end
            CoefGroupIx = cell(nlevel+1,1 );
            CoefGroupIx{1} = cell2mat(Aix)';
            for i=1:nlevel
                CoefGroupIx{i+1} = reshape( [ Dix{:,i}],1,[]);
            end
        end

        function y=mult(obj,x)
            if ~all(isreal(x) )
                y = obj.mult(real(x)) + j*obj.mult(imag(x));
                return
            end

            if obj.opts.scale ~= 1
                x = x * obj.opts.scale;
            end
            imdim = obj.S{1}(end,:);
            im = reshape(x,imdim(1),imdim(2));
            y=zeros(obj.dims(1),1);

            offset=0; % track offset into y vector
            for k=1:obj.numBases
                C = wavedec2(im,obj.nlevel,obj.wname{k});
                if obj.skips(k) >0
                    C = C( (obj.skips(k)+1):end);
                end
                y(offset+(1:length(C))) = C;
                offset = offset + length(C);
            end
        end

        function x = multTr(obj,y)
            if ~all(isreal(y) )
                x = obj.multTr(real(y)) + j*obj.multTr(imag(y));
                return
            end
            x = zeros(obj.dims(2),1);
            offset=0; % track offset into y vector
            for k=1:obj.numBases
                npad = obj.skips(k);
                yp = y( offset+(1:(obj.ncoef(k)-npad) ));
                if npad>0
                    yp = [ zeros(npad,1); yp(:)];
                end
                x = x + reshape( waverec2( yp ,obj.S{k} , obj.wname{k} ) ,[],1 );
                offset = offset + obj.ncoef(k) - npad;
            end
            if obj.opts.scale ~= 1
                x = x * obj.opts.scale;
            end
        end
    end
end
