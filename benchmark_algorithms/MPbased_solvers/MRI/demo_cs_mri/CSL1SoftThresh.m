function x = CSL1SoftThresh(x0,param)

% 
% res = CSL1SoftThresh(x0,param)
%
% Compressed sensing reconstruction of undersampled k-space MRI data
%
% L1-norm minimization using iterative soft-thresholding
% 
% Given the acquisition model y = E*x and the sparsifying transform W, the
% program perfoms soft-thresholding on W*x to estimate the missing values
% in the k-apce data y
%
% Ricardo Otazo 2009
%
x=x0;
ite=0;

% iterations
while(1)
    x0=x;
    X=param.W*x0;
    %ttt=thselect(X(:),'minimaxi')./100
    ttt=param.lambda;
    X = SoftThresh(X,ttt);
    x=param.W'*X;
    Y = param.FT*x;
    Y = Y.*(param.y==0) + param.y;
    x = param.FT'*Y;
    ite = ite + 1;
    % print some numbers for debug purposes	
    if param.display,
        disp(sprintf('%d   , obj: %f, ||x_1-x_0||/||x_0||: %f', ite,norm(X(:),1),norm(x(:)-x0(:))/norm(x0(:))));
        %figure(100),aux(:,:,1,:)=abs(x);montage(aux,[0,0.75]);title(strcat('JOCS, ite: ',int2str(ite)));drawnow
    end
    % stopping criteria 
	if (ite > param.nite) || (norm(x(:)-x0(:))<param.tol*norm(x0(:))), break;end
    
end
return;

function y=SoftThresh(x,p)
y=(abs(x)-p).*x./abs(x).*(abs(x)>p);
    