function ser = measSER(xhatTot, x, x0)
% measSER:  Measure the symbol error rate 
%
% Let indTrue(i) = index k of the constellation point in x0(k) closest
% to x(i). Similarly let indEst(i,j) = index k of the constellation 
% point in x0(k) closest to xhatTot(i,j).  The symbol error rate is 
% defined as 
%   ser(j) = mean number of values i such that indEst(i,j) ~= indTrue(i)

% Get dimensions
[nx,nt] = size(xhatTot);
nx0 = length(x0);

% Find closest constellation points of true value
dx = abs(repmat(x,1,nx0) - repmat(x0',nx,1));
[mm,indTrue] = min(dx,[],2);
    
% Compute SER
ser = zeros(nt,1);
for it=1:nt
    % Find closest constellation points of estimate
    dx = abs(repmat(xhatTot(:,it),1,nx0) - repmat(x0',nx,1));
    [mm,indEst] = min(dx,[],2);
    
    % Compute SER
    ser(it) = mean(indTrue ~= indEst);
       
end

end