% LinTransSubset: choose a subset of rows and/or columns from a linear operator
classdef LinTransSubset < LinTrans

    properties
        trans; % the wrapped LinTrans object
        ri; % row indices
        ci; % col indices
    end
    methods
        function obj = LinTransSubset(ltop,ri,ci)
            [m,n] = ltop.size();
            if numel(ri)==0
                ri = 1:m;
            end
            if nargin < 3 || numel(ci) ==0
                ci = 1:n;
            end
            obj = obj@LinTrans( length(ri) , length(ci) );

            obj.ri = ri;
            obj.ci = ci;
            obj.trans = ltop;
        end

        function y=mult(obj,x)  
            [m,n] = obj.trans.size();
            xf = zeros( n,size(x,2) );
            xf(obj.ci,:) =x;
            y = obj.trans.mult( xf );
            y = y(obj.ri,:);
        end

        function x=multTr(obj,y)
            [m,n] = obj.trans.size();
            yf = zeros( m,size(y,2) );
            yf(obj.ri,:) = y;
            x = obj.trans.multTr(yf);
            x = x(obj.ci,:);
        end

        function y=multSq(obj,x)
            [m,n] = obj.trans.size();
            xf = zeros( n,size(x,2) );
            xf(obj.ci,:) =x;
            y = obj.trans.multSq( xf );
            y = y(obj.ri,:);
        end

        function x=multSqTr(obj,y)
            [m,n] = obj.trans.size();
            yf = zeros( m,size(y,2) );
            yf(obj.ri,:) = y;
            x = obj.trans.multSqTr(yf);
            x = x(obj.ci,:);
        end
    end
end
