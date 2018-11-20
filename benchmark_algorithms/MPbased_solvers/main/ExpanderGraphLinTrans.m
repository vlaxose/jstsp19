classdef ExpanderGraphLinTrans < LinTrans
    % ExpanderGraphLinTrans:  Expander Graph linear transform implemented
    % using sparse matrix. The matrix is the adjacency matrix of the
    % expander graph. This class just generates the columns randomly.
    % It is intended to be sub-classed to implement particular EG
    % constructions.
    
    %Matrix is stored as private to prevent user changes
    properties (SetAccess = private) 
        A;      % Adjacency matrix of the expander graph (sparse)
    end
    
    methods
        
        % Constructor. Creates an M by N expander graph with P 1's per
        % column
        function obj = ExpanderGraphLinTrans(M,N,P)
            
            %Call super init
            obj = obj@LinTrans;
            
            %Allocate memory for sparse matrix
            obj.A = spalloc(M,N,N*P);
            
            %Build an adjacnecy matrix for a random graph with exactly P
            %edges to each variable node
            for n = 1:N
                
                %Set random positions
                whichLocs = [];
                while length(whichLocs) ~= P
                    whichLocs = unique(randi(M,P,1));
                end
                
                %Assign the ones
                obj.A(whichLocs,n) = 1;
                
            end
            
        end
        
        % Size
        function [m,n] = size(obj)
            [m,n] = size(obj.A);
        end
        
        % Matrix multiply
        function y = mult(obj,x)
            y = obj.A*x;
        end
        % Matrix multiply transpose
        function y = multTr(obj,x)
            y = obj.A'*x;
        end
        % Matrix multiply with square- same matrix, all entries 0/1
        function y = multSq(obj,x)
            y = obj.A*x;
        end
        % Matrix multiply transpose - same matrix, all entries 0/1
        function y = multSqTr(obj,x)
            y = obj.A'*x;
        end
        
        
    end
end