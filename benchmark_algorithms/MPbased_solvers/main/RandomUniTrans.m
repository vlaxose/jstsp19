classdef RandomUniTrans < LinTrans
    % RandomUniTrans performs a unitary transform in O(NlogN) time.
    % The matrix form (although never explicitly computed)
    % has elements that mimic a Gaussian distribution.
    % 
    % examples:
    %   rut = RandomUniTrans(m);
    %   y = rut * x;   % rut acts upon x like a mxm matrix with Gaussian-like entries and orthonormal columns/row
    %   x2 = rut' * y; % rut' is inverse of rut
    %   rut2 = RandomUniTrans(m,rut.plan);  % the plan is needed for reproducibility
    %
    % for a demo of the properties, run : RandomUniTrans.selftest(64,1e3) 
    %
    %  Mark Borgerding ( borgerding.7 osu edu ) 2013-Apr-2

    properties
        plan % cell array to hold the small block matrices and the permutation indices for each iteration
    end

    methods
        function obj = RandomUniTrans(m,plan)
            obj = obj@LinTrans(m,m, sqrt(m) );
            if rem(m,2) 
                 error('RandomUniTrans only available for even sizes (FIXME)');
            end
            if nargin < 2
              plan.radix = 16;  % the dimension of the subblock matrices
              plan.nit = 3*round(log(m)/log(plan.radix)); % several passes of mixing gives nice Gaussian-like properties
              plan.ix = zeros(m,plan.nit); 
              plan.submat = cell(plan.nit,1);
              for k=1:plan.nit
                  plan.ix(:,k) = randperm(m);       % permutation indices 
                  plan.submat{k} =  orth(randn(plan.radix)); % orthobase for the small block
              end
            end
            if rem(m,plan.radix) 
                error(sprintf('m=%d not divisible by radix %d (FIXME)',m,plan.radix));
            end
            obj.plan = plan;
        end

        function y = mult(obj,x)
            for k=1:obj.plan.nit
                x = reshape( obj.plan.submat{k} * reshape( x( obj.plan.ix(:,k),: ),obj.plan.radix,[]),obj.dims(1),[]);
            end
            y=x;
        end

        function y = multTr(obj,x)
            for k=obj.plan.nit:-1:1
                x( obj.plan.ix(:,k),: ) = reshape( obj.plan.submat{k}' * reshape(x(:),obj.plan.radix,[]),obj.dims(1),[]);
            end
            y=x;
        end
    end

    methods (Static)
        function failed = selftest(m,ntrials)
            if nargin < 1
                m=64;
            end
            if nargin <2
                ntrials=1e2;
            end
            failed = [];
            meanvals = zeros(ntrials,2);
            varvals=zeros(ntrials,2); 
            svals=zeros(ntrials,2);
            kvals=zeros(ntrials,2); 
            z=linspace(-6,6,151)';

            dz = z(2)-z(1);
            H1 = 0*z;
            H2 = 0*z;
            for k=1:ntrials
                rut = RandomUniTrans(m);
                RUT = rut * eye(m);
                r1 = RUT(:)*sqrt(m); % should be normally distributed
                r2=randn(m^2,1);
                %r1 = r1(1:(length(r1)/2));
                %r2 = r2(1:(length(r2)/2));
                meanvals(k,:) = [ mean(r1 ) mean(r2) ]';
                varvals(k,:) = [var(r1 ) var(r2) ]';
                svals(k,:) = [skewness(r1 ) skewness(r2) ]';
                kvals(k,:) = [ kurtosis( r1) kurtosis(r2) ]';

                x = randn(m,m);
                y = rut * x;
                x2 = rut' * y;
                if max(norm( RUT*RUT' - eye(m)),norm( RUT'*RUT - eye(m))) > 1e-9 ||  norm( x2 - x ) > 1e-9
                    fprintf(2,'failed unitary test');
                    failed = rut;
                    return
                end
                H1 = H1 + hist(r1,z)'/length(r1)/dz;
                H2 = H2 + hist(r2,z)'/length(r2)/dz;
            end
            H1 = H1 / ntrials;
            H2 = H2 / ntrials;

            npdf = normpdf(z(:));
            nzi = find(H1);
            D_kl1 = sum( npdf(nzi) .* log (npdf(nzi) ./ H1(nzi) ) )*dz;
            nzi = find(H2);
            D_kl2 = sum( npdf(nzi) .* log (npdf(nzi) ./ H2(nzi) ) )*dz;
            fprintf('KL divergence of RUT=%.3g , of randn=%.3g\n',D_kl1,D_kl2);

            figure(1); clf reset
            % scale for imshow so -3 sigmas is zero and +3 sigmas is 1
            imshow(RUT*sqrt(m)/6 +.5 );title('typical RUT "matrix"')

            figure(2); clf reset
            plot(z,[normpdf(z(:)) H1(:) H2(:) ] );
            legend('Normal pdf','RUT element (scaled) histogram','randn histogram')

            figure(3); clf reset
            t=linspace(0,1,ntrials);
            subplot(221); plot(t,sort(meanvals)); legend('RUT','randn'); title('sorted mean of each trial')
            subplot(222); plot(t,sort(varvals)); legend('RUT','randn'); title('variance')
            subplot(223); plot(t,sort(svals)); legend('RUT','randn'); title('skewness')
            subplot(224); plot(t,sort(kvals)); legend('RUT','randn'); title('kurtosis')
        end
    end
end
