function y =Projection(Pt,b,T)

if nargin < 3

y = b - Pt*(Pt'*b);

else
    
It = speye(size(Pt,1));
It = It(:,T);
y = It*b - Pt*(Pt(T,:)'*b);

end
