classdef LinTransConcat < LinTrans 
    properties
        lta; % cell array of LinTrans objects
        sz; % size of result
        scales;
    end

    methods 
         % Constructor 
         % linTransArray is a cellular array. row or column orientation 
         % dictates the orientation of the concatenation
         % scaling is optional
         function obj = LinTransConcat(linTransArray,scaling) 
            obj = obj@LinTrans;
            obj.lta = linTransArray;
            if ( ~iscell(obj.lta) )
                error('LinTransConcat constructor require cell array');
            end

            % convert numeric matrices to MatrixLinTrans objects if needed
            for k=1:length(obj.lta)
                if isobject( obj.lta{k} ) && ismethod(obj.lta{k},'multSqTr')
                    ;% this object appears to be a usable object already 
                else
                    obj.lta{k} = MatrixLinTrans(obj.lta{k} );
                end
            end

            if nargin<2
               scaling =  ones(size(obj.lta) );
            end
            if iscell(scaling)  
                scaling = cell2mat(scaling);
            end
            obj.scales = scaling;

            if size(obj.lta,1)==1
                growDim = 2; % growing horizonally (increase length of LinTrans input)
            elseif size(obj.lta,2)==1
                growDim = 1; % growing vertically (increase length of LinTrans output)
            else
                error('LinTransConcat can only concatenate horizontally or vertically, not both');
            end
            ngDim=3-growDim;% nongrowing dimension

            [m,n] = size(obj.lta{1});
            obj.sz=[m,n];
            for k=2:length(obj.lta)
                [m,n] = size( obj.lta{k} );
                szk = [m,n];
                if szk(ngDim) ~= obj.sz(ngDim)
                    error ([ 'incompatible sizes for concatenation in dim ' num2str(ngDim)]);
                end
                obj.sz(growDim) = obj.sz(growDim) + szk(growDim);
            end
         end

        function [m,n] = size(obj,dim)
            if nargin>1
                if dim>2
                    m=1;
                else
                    m=obj.sz(dim); 
                end
            elseif nargout<2
                m=obj.sz;
            else
                m=obj.sz(1);
                n=obj.sz(2);
            end
        end

        % Matrix multiply:  z = A*x
        function y = mult(obj,x)
            y=zeros(obj.sz(1),size(x,2));
            i=1;
            for k=1:length(obj.lta)
                s = obj.scales(k);
                [m,n] = size( obj.lta{k} );
                if size(obj.lta,1)==1 % (increase length of LinTrans input)
                    y = y + s * obj.lta{k}.mult( x(i:i+n-1,:) );
                    i=i+n;
                else % growing vertically (increase length of LinTrans output)
                    y(i:i+m-1,:) = s * obj.lta{k}.mult( x );
                    i=i+m;
                end
            end
        end
        % Matrix multiply with square:  z = (A.^2)*x
        function z = multSq(obj,x)
            z=zeros(obj.sz(1),size(x,2));
            i=1;
            for k=1:length(obj.lta)
                s = obj.scales(k)^2;
                [m,n] = size( obj.lta{k} );
                if size(obj.lta,1)==1 % (increase length of LinTrans input)
                    z = z +  s * obj.lta{k}.multSq( x(i:i+n-1,:) );
                    i=i+n;
                else % growing vertically (increase length of LinTrans output)
                    z(i:i+m-1,:) = s *  obj.lta{k}.multSq( x );
                    i=i+m;
                end
            end
        end


        % Matrix multiply transpose:  x = A'*z
        function x = multTr(obj,z)
            x=zeros(obj.sz(2),size(z,2) );
            i=1;
            for k=1:length(obj.lta)
                s = obj.scales(k);
                [m,n] = size( obj.lta{k} );
                if size(obj.lta,1)==1 
                    x(i:i+n-1,:) = s * obj.lta{k}.multTr(z);
                    i=i+n;
                else 
                    x = x + s * obj.lta{k}.multTr( z(i:i+m-1,:) );
                    i=i+m;
                end
            end
        end

        % Matrix multiply with componentwise square transpose:  
        % x = (A.^2)'*z
        function x = multSqTr(obj,z)   
           x=zeros(obj.sz(2),size(z,2) );
           i=1;
           for k=1:length(obj.lta)
               s = obj.scales(k)^2;
               [m,n] = size( obj.lta{k} );
               if size(obj.lta,1)==1 
                   x(i:i+n-1,:) = s * obj.lta{k}.multSqTr(z);
                   i=i+n;
               else 
                   x = x +  s * obj.lta{k}.multSqTr( z(i:i+m-1,:) );
                   i=i+m;
               end
           end
        end
    end
end
