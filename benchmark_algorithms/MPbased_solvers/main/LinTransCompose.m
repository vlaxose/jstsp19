classdef LinTransCompose < LinTrans
    % LinTransCompose: a composition of linear operators
    % 
    % The "outermost" linear operator (or the last to be applied) is the first in the cell array.
    % e.g. if you want to compose (A1*A2*A3), then
    % use LinTransCompose({A1,A2,A3})
    %
    %  Mark Borgerding ( borgerding.7 osu edu ) 2013-Apr-2
    properties
        lt; % cell array of LinTrans objects
    end

    methods 
         % Constructor 
         % The last LinTrans in the array is the first one applied to the 
         function obj = LinTransCompose(linTransArray) 
            if ( ~iscell(linTransArray) || length(linTransArray) == 0 )
                error('LinTransCompose constructor require cell array');
            end

            % determine sizes and check dimensional compatibility
            [mPrev,nPrev] = size(linTransArray{1});
            myRows = mPrev;
            for k = 2:length(linTransArray)
                [m,n] = size(linTransArray{k});
                if nPrev ~= m
                    error(sprintf('invalid composition: %dx%d with %dx%d',mPrev,nPrev,m,n));
                end
                mPrev = m;
                nPrev = n;
            end
            myCols = nPrev;

            obj = obj@LinTrans(myRows,myCols,0);  % let LinTrans figure out the overall scaling for the mulSq and multSqTr
            obj.lt = linTransArray;
         end

        % Matrix multiply:  z = A1*A2*...An *x
        function z = mult(obj,x)
            for k=length(obj.lt):-1:1
                x = obj.lt{k} * x;
            end
            z=x;
        end

        % Matrix multiply transpose:  x = An'*...A2'*A1*z
        function x = multTr(obj,z)
            for k=1:length(obj.lt)
                z = obj.lt{k}' * z;
            end
            x=z;
        end
    end
end
