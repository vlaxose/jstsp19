classdef LinTransTV < LinTrans
    % Perform 1d or 2d difference operator
    % The operator is scaled to unit norm rows so norm( lt * x)/norm(x) =~ 1
    %
    % 1d example: for a n-by-1 vector x, the following y0,y1 are equivalent (n-1)-by-vectors
    %       y0 = -sqrt(.5)*diff(x);
    %       y1 = LinTransTV(n) * x;
    %
    % 2d example: for a m-by-n image x, the following y0,y1 are equivalent q-by-1 vectors (q=2mn-m-n)
    %       y0 = -sqrt(.5) * [ reshape( diff(x')',[],1)   ; reshape(diff(x),[],1) ];
    %       y1 = LinTransTV(size(x)) * x(:); 
    %
    % Important note: 2d images must currently be flattened to a vector space a la x(:) 
    % 
    properties 
        D;% sparse matrix with two non-zero elements per row: sqrt(.5) and -sqrt(.5)
        Dsq; % D.^2
    end
    % Implementation note:
    % The difference operator is currently implemented with sparse matrices.
    % It might be faster to use indexing operations, but it is not obvious how to
    % implement the 2d adjoint operator simply.

    methods
        % constructor    
        function obj = LinTransTV(dims,scaling)
            ndims = length(dims);
            nx = prod(dims);
            if ndims==1
                I = speye(nx-1);
                P = spalloc(nx-1,1,0);
                D = sqrt(.5) *[ [I P] - [P I] ];
            elseif ndims==2
                d1 = dims(1); 
                d2 = dims(2);

                % horizontal compares
                nh = (d2-1)*d1;
                Ih = speye(nh);
                Ph = spalloc(nh,nx-nh,0);
                Dh = [Ih Ph] - [Ph Ih];

                % vertical compares 
                nvp = d1*d2-1;
                Iv = speye(nvp);
                Pv = spalloc(nvp,1,0);
                Dvp = [Iv Pv] - [Pv Iv];
                ix = find(rem(1:nx,d1));
                Dv = Dvp(ix,:);
                D =  sqrt(.5)*[Dh;Dv];
            end
            obj = obj@LinTrans( size(D,1) ,prod(dims) );
            if nargin>1
                D = D * scaling;
            end
            obj.D = D;
            obj.Dsq = D.^2;
        end

        function y=mult(obj,x)
            y = obj.D* x;
        end
        function x=multTr(obj,y)
            x = obj.D' * y;
        end
        function y=multSq(obj,x)
            y = obj.Dsq* x;
        end
        function x=multSqTr(obj,y)
            x = obj.Dsq' * y;
        end
    end
end
