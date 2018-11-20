classdef LinTransDiag < LinTrans
    % LinTransDiag: Represent a block diagonal linear transformation.
    % Dimensions are analogous to the blkdiag(...) MATLAB command

    properties
        lt; % cell array of LinTrans objects, in order along the diagonal
    end

    methods 
         % Constructor  
         function obj = LinTransDiag(linTransArray) 
            if ( ~iscell(linTransArray) || isempty(linTransArray) )
                error('LinTransDiag constructor require cell array');
            end

            %Determine size
            myRows = 0;
            myCols = 0;
            for kk = 1:length(linTransArray)
                [m,n] = size(linTransArray{kk});
                myRows = myRows + m;
                myCols = myCols + n;
            end

            obj = obj@LinTrans(myRows,myCols,0);  % let LinTrans figure out the overall scaling for the mulSq and multSqTr
            obj.lt = linTransArray;
         end

        % Matrix multiply
        function y = mult(obj,x)
            
            %Preallocate
            [M,~] = obj.size();
            L = size(x,2);
            y = zeros(M,L);
            
            startLocM = 1;
            startLocN = 1;
            for k=1:length(obj.lt)
                %Add length of previous operator
                if k > 1
                    startLocM = startLocM + m;
                    startLocN = startLocN + n;
                end
                
                %Get size of current operator
                [m,n] = size(obj.lt{k});
                
                %Do the mult
                y(startLocM:(startLocM+m-1),:) =...
                    obj.lt{k}*x(startLocN:(startLocN+n-1),:);

            end

        end

        % Matrix multiply transpose
        function y = multTr(obj,x)
            
           %Preallocate
            [~,N] = obj.size();
            L = size(x,2);
            y = zeros(N,L);
            
            startLocM = 1;
            startLocN = 1;
            for k=1:length(obj.lt)
                %Add length of previous operator
                if k > 1
                    startLocM = startLocM + m;
                    startLocN = startLocN + n;
                end
                
                %Get size of current operator
                [m,n] = size(obj.lt{k});
                
                %Do the mult
                y(startLocN:(startLocN+n-1),:) =...
                    obj.lt{k}'*x(startLocM:(startLocM+m-1),:);

            end
        end
    end
end
