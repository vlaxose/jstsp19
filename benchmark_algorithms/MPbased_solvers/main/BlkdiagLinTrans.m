classdef BlkdiagLinTrans < LinTrans
    % BlkdiagLinTrans:  Linear transform class with a block diagonal matrix
    % This simple version assumes that the square blocks are all equal
    properties
        bmat;   % the block matrix
        bmatsq; % Asq = (bmat.^2)
        Nblocks; %Number of blocks
    end
    
    methods
        
        % Constructor
        function obj = BlkdiagLinTrans(bmat,Nblocks)
            obj = obj@LinTrans;
            obj.bmat = bmat;
            obj.bmatsq = (abs(bmat).^2);
            obj.Nblocks = Nblocks;
            
        end
        
        % Size
        function [m,n] = size(obj)
            m = obj.Nblocks*size(obj.bmat,1);
            n = m;
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            mm = size(obj.bmat,1);
            y = reshape(obj.bmat*reshape(x,mm,[]),obj.Nblocks*mm,[]);
        end
        % Matrix multiply transpose
        function y = multTr(obj,x)
            mm = size(obj.bmat,1);
            y = reshape(obj.bmat'*reshape(x,mm,[]),obj.Nblocks*mm,[]);
        end
        % Matrix multiply with square
        function y = multSq(obj,x)
            mm = size(obj.bmat,1);
            y = reshape(obj.bmatsq*reshape(x,mm,[]),obj.Nblocks*mm,[]);
        end
        % Matrix multiply transpose
        function y = multSqTr(obj,x)
            mm = size(obj.bmat,1);
            y = reshape(obj.bmatsq'*reshape(x,mm,[]),obj.Nblocks*mm,[]);
        end
        
        
        
    end
end