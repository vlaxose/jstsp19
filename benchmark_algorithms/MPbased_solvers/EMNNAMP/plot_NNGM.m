% Given the parameters of a non-negative Gaussian-Mixture, plot_NNGM plots 
% it to give the user a visual understanding of the pdf (NN Gaussian 
% Mixture) and the pmf (Bernoulli).  Specifially, the Gaussian Mixture is modeled as:
%
%   p(X(n,t)) = (1-tau(t))*delta(X(n,t))
%      + sum_k^L tau(t)*omega(t,k) (C) normal_+(X(n,t);theta(t,k),phi(t,k)
%
% where tau, omega, theta, phi are the sparsity, mixture weights, 
% locations, and scales. (The kth component of the vectors correspond the 
% the kth element). plot_NNGM also allows for complex pdfs, in which the output
% subplots the pdf on the real and imaginary axis.  For all parts the pdf
% scale is on the left y axis, and the pmf scale is on the right y axis.
%
% SYNTAX                    [tag1, tag2, AX] = plot_NNGM(param,color,linestyle)
%
% INPUT
%   -params                 Structure of Gaussian Mixture parameters
%       .tau             Sparsity level
%       .active_weights     weights of GM (omega)
%       .active_mean        Means of active components (theta) 
%       .active_var         Variances of active components (phi)
%   -color                  color in string format for plotting of real pdfs
%                           [example 'r' for red]
%   -linestyle              linestlye in string format for plotting of real
%                           pdfs. [example '-' for solid]
%
%   NOTE: param must have GM parameters of length L by 1 and a scalar
%   tau.  If not, plot_GM assumes the first L by 1 vector only.
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
%
% Version 1.0
%%
function [tag1, tag2, AX] = plot_NNGM(param,color,linestyle)

if nargin < 2
    color = 'b';
end

if nargin < 3
    linestyle = '-';
end

[L, T] = size(param.active_loc);
if ~isfield(param,'active_weights')
    param.active_weights = 1;
end
    
if T > 1
    param.tau = param.tau(:,1);
    param.active_weights = param.active_weights(:,1);
    param.active_loc = param.active_loc(:,1);
    param.active_scales = param.active_scales(:,1);
end

%Find appropriate minimum and maximum values for domain
[min_x, temp] = min(param.active_loc);
min_x = min(-1, min_x - 5*sqrt(param.active_scales(temp)));
[max_x, temp] = max(param.active_loc);
max_x = max(1, max_x + 5*sqrt(param.active_scales(temp)));

x = linspace(min_x,max_x,256);
y = zeros(1,256);

%Calculate active pdf
for i = 1:L
   y = y + param.active_weights(i)*exp(-(x-param.active_loc(i)).^2....
    ./(2.*param.active_scales(i)))./sqrt(2*pi*param.active_scales(i))...
    ./(0.5*erfc(-param.active_loc(i)./sqrt(param.active_scales(i)*2)));
end
y(x <0) = 0;
dx = x(2) - x(1);
y = y./(sum(y)*dx).*param.tau;

[AX,tag1,tag2] = plotyy(x,y , 0, 1-param.tau,'plot', 'stem');
grid on

set(get(AX(1),'Ylabel'),'String','pdf (x)') 
set(get(AX(2),'Ylabel'),'String','pmf (x)') 

set(tag1,'linewidth',2,'Color',color,'LineStyle',linestyle)
set(tag2,'linewidth',2,'Color',color,'LineStyle',linestyle)
    


return