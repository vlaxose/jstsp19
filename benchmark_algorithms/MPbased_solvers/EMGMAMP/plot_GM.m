% Given the parameters of a Gaussian-Mixture, plot_GM plots it to give the 
% user a visual understanding of the pdf (Gaussian Mixture) and the pmf 
% (Bernoulli).  Specifially, the Gaussian Mixture is modeled as:
%
%   p(X(n,t)) = (1-lambda(t))*delta(X(n,t))
%      + sum_k^L lambda(t)*omega(t,k) (C) normal(X(n,t);theta(t,k),phi(t,k)
%
% where lambda, omega, theta, phi are the sparsity, mixture weights, 
% means, and variances. (The kth component of the vectors correspond the the
% lth element). plot_GM also allows for complex pdfs, in which the output
% subplots the pdf on the real and imaginary axis.  For all parts the pdf
% scale is on the left y axis, and the pmf scale is on the right y axis.
%
% SYNTAX                    [tag1, tag2, AX] = plot_GM(param,color,linestyle)
%
% INPUT
%   -params                 Structure of Gaussian Mixture parameters
%       .lambda             Sparsity level
%       .active_weights     weights of GM (omega)
%       .active_mean        Means of active components (theta) 
%       .active_var         Variances of active components (phi)
%   -color                  color in string format for plotting of real pdfs
%                           [example 'r' for red]
%   -linestyle              linestlye in string format for plotting of real
%                           pdfs. [example '-' for solid]
%
%   NOTE: param must have GM parameters of length L by 1 and a scalar
%   lambda.  If not, plot_GM assumes the first L by 1 vector only.
%
% OUTPUT
%   -tag1                   A tag for the plot of the pdf
%   -tag2                   A tag for the plot of the pmf (Bernoulli component)
%   -AX                     The axes tags for the pdf and pmf                 
%
% Coded by: Jeremy Vila, The Ohio State Univ.
% E-mail: vilaj@ece.osu.edu
% Last change: 4/4/12
% Change summary: 
%   v 1.0 (JV)- First release
%   v 2.0 (JV)- Can plot complex pdfs as well.
%
% Version 2.0
%%
function [tag1, tag2, AX] = plot_GM(param,color,linestyle)

if nargin < 2
    color = 'b';
end

if nargin < 3
    linestyle = '-';
end

[L T] = size(param.active_mean);
if ~isfield(param,'active_weights')
    param.active_weights = 1;
end
    
if T > 1
    param.lambda = param.lambda(:,1);
    param.active_weights = param.active_weights(:,1);
    param.active_mean = param.active_mean(:,1);
    param.active_var = param.active_var(:,1);
end

if isreal(param.active_mean)
    %Find appropriate minimum and maximum values for domain
    [min_x, temp] = min(param.active_mean);
    min_x = min(-1, min_x - 5*sqrt(param.active_var(temp)));
    [max_x, temp] = max(param.active_mean);
    max_x = max(1, max_x + 5*sqrt(param.active_var(temp)));

    x = linspace(min_x,max_x,256);
    y = zeros(1,256);

    %Calculate active pdf
    for i = 1:L
       y = y + param.lambda*param.active_weights(i)*exp(-(x-param.active_mean(i)).^2....
        ./(2.*param.active_var(i)))./sqrt(2*pi*param.active_var(i));
    end

    %subplot(1,1,1)
    [AX,tag1,tag2] = plotyy(x,y , 0, 1-param.lambda,'plot', 'stem');

    set(get(AX(1),'Ylabel'),'String','pdf (x)') 
    set(get(AX(2),'Ylabel'),'String','pmf (x)') 

    set(tag1,'linewidth',2,'Color',color,'LineStyle',linestyle)
    set(tag2,'linewidth',2,'Color',color,'LineStyle',linestyle)
    
else
    %Find appropriate minimum and maximum values for domain
    [min_realx, temp] = min(real(param.active_mean));
    min_realx = min(-1, min_realx - 3*sqrt(param.active_var(temp)));
    [max_realx, temp] = max(real(param.active_mean));
    max_realx = max(1, max_realx + 3*sqrt(param.active_var(temp)));
    
    [min_imagx, temp] = min(imag(param.active_mean));
    min_imagx = min(-1, min_imagx - 3*sqrt(param.active_var(temp)));
    [max_imagx, temp] = max(imag(param.active_mean));
    max_imagx = max(1, max_imagx + 3*sqrt(param.active_var(temp)));

    x = linspace(min_realx,max_realx,256);
    y = linspace(min_imagx,max_imagx,256);
    z1 = zeros(1,256);
    z2 = zeros(1,256);

    %Calculate active pdf
    for i = 1:L
      z1 = z1 + param.lambda*param.active_weights(i)*exp(-(x-real(param.active_mean(i))).^2....
        ./param.active_var(i))./(2*pi*param.active_var(i));
      z2 = z2 + param.lambda*param.active_weights(i)*exp(-(y-imag(param.active_mean(i))).^2....
        ./param.active_var(i))./(2*pi*param.active_var(i));
    end
    
    subplot(2,1,1)
    [AX,tag1,tag2] = plotyy(x, z1 , 0, 1-param.lambda,'plot', 'stem');

    set(get(AX(1),'Ylabel'),'String','pdf (x)') 
    set(get(AX(2),'Ylabel'),'String','pmf (x)') 

    set(tag1,'linewidth',2,'Color',color,'LineStyle',linestyle)
    set(tag2,'linewidth',2,'Color',color,'LineStyle',linestyle)
    xlabel('real(x)')
    title('Estimated pdf/pmf for the real axis')
    
    subplot(2,1,2)
    [AX,tag1,tag2] = plotyy(y, z2 , 0, 1-param.lambda,'plot', 'stem');

    set(get(AX(1),'Ylabel'),'String','pdf (x)') 
    set(get(AX(2),'Ylabel'),'String','pmf (x)') 

    set(tag1,'linewidth',2,'Color',color,'LineStyle',linestyle)
    set(tag2,'linewidth',2,'Color',color,'LineStyle',linestyle)
    xlabel('imag(x)')
    title('Estimated pdf/pmf for the imaginary axis')
end

if nargout==0, clear tag1; clear tag2; clear Ax; end;

return
