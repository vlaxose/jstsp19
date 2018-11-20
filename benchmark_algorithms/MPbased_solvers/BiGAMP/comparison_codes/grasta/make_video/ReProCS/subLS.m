function xhat = subLS(Pt,T,y)

%A = Pt(T,:);

It = speye(size(Pt,1));
It = It(:,T);
C = It - Pt*Pt(T,:)';

% xhat = pinv(C)*y;
%xhat = C\y;
% xhat = lsqr(C,y,1e-3,50);
[xhat,~] = lsqr(C,y);