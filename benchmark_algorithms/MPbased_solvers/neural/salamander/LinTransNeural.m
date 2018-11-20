% LinTrans:  Linear transform class with a matrix
classdef LinTransNeural < handle
    properties
        A;      % Binary packed matrix A
        Irow;   % Rows to output
        ndly;   % Number delay
        nin;    % Number of inputs        
        nx;     % Total number of unknown weights = ndly*nin
    end
    
    methods
        
        % Constructor
        function obj = LinTransNeural(A,Irow,ndly)
            obj.A = A;
            obj.Irow = Irow; 
            obj.ndly = ndly;
            obj.nin = A.ncol;
            obj.nx = obj.ndly * obj.nin;
        end
        
        % Size
        function [m,n] = size(obj)
            m = length(obj.Irow);
            n = obj.nx + 1;
        end
        
        % Matrix multiply
        function z = mult(obj,u)
            xsq = reshape(u(1:obj.nx),obj.ndly, obj.nin);
            z = obj.A.firFilt(xsq, obj.Irow) + u(obj.nx+1);
        end
        % Matrix multiply transpose
        function u = multTr(obj,z)
            xsq = obj.A.firFiltTr(z, obj.ndly, obj.Irow);
            x = xsq(:);
            z0 = sum(z);
            u = [x; z0];
        end
        % Matrix multiply with square
        function z = multSq(obj,u)
            z = obj.mult(u);
        end
        % Matrix multiply transpose
        function u = multSqTr(obj,z)
            u = obj.multTr(z);
        end        
        
    end
end