% out = s_transform(in,eig,N,[opt]) 
%
% Evaluates the S-transform of an NxN PSD Hermitian matrix whose 
% eigenvalues are specified in the vector "eig".  This approach uses 
% the eta-transform from Tulino/Verdu book.  In particular, it computes
%   S(y) = -(y+1)./y .* eta_inv(1+y)
% where
%   eta_inv(1+y) is the value of gamma such that eta(gamma)=1+y
% and
%   eta(gamma) = mean( 1./(1+[eig(:);zeros(N-R,1)].*gam) ) .
%   R = rank of matrix = sum(eig>0)
%
% Since eta(.) takes values between 1 and (N-R)/N, an error is 
% reported if any value of y is outside the range [-R/N,0].

function out = s_transform(in,eig,N,varargin)

% set default options
opt.type = 'newton';
opt.nit = 50;
opt.tol = 1e-6;
opt.debug = false;
opt.verbose = false;

% overwrite options 
for i = 1:2:length(varargin)
  opt.(varargin{i}) = varargin{i+1};
end

% check eigs
if any(eig<0)||(length(eig)>N)
  error('2nd input must be a non-negative vector of length <= N')
end

% process singular values
eig = [eig(:);zeros(N-length(eig),1)]; % zero-pad to length N
R = sum(eig>0); % rank
eig_mean = mean(eig);
eig_min = min(eig(1:R));

% check input to make sure it is feasible
if any(in(:)<-R/N)||any(in(:)>0)
  error('1st input must be in interval [-R/N,0]=[%.4f,0]',-R/N)
end

% define eta-transform
etatr = @(gam) mean(1./(1+eig*gam),1); % eta transform, handles row vector gam

% plot if debugging
if opt.debug
  etatr_ub = @(gam) mean(1./eig(1:R))./gam + (N-R)/N; % upper bound on etatr
  etatr_lb = @(gam) 1./(1+eig_mean*gam); % lower bound on etatr 
  gam_high = mean(1./eig(1:R))/(in(1)+R/N);
  gam_grid = linspace(0,1.1*gam_high,1e4);

  clf;
  plot(gam_grid,etatr(gam_grid));
  hold on
  plot(gam_grid,etatr_ub(gam_grid),'--');
  plot(gam_grid,etatr_lb(gam_grid),'--');
  plot([min(gam_grid),max(gam_grid)],[1,1]*(1+in(1)),':')
  axis([min(gam_grid),max(gam_grid),0,2])
  legend('eta transform','upper bound','lower bound','target')
  xlabel('gamma')
  ylabel('eta transform')
end

% handle inputs equal to zero and -R/N
out = nan(size(in)); 
indx0 = find(in==0); out(indx0) = 1;
indx1 = find(in==-R/N); out(indx1) = inf;

% handle non-zero-valued inputs
nz = find(~(in==0 | in==-R/N)); % indices of non-zero inputs

switch opt.type
  case 'bisection'

    for l=1:length(nz) % for each non-zero input...

      % compute inverse eta-transform
      gam_low = (1/(1+in(nz(l))) -1)/eig_mean;
      gam_high = mean(1./eig(1:R))/(in(nz(l))+R/N);
      gam = 0.5*(gam_low+gam_high);
      for t=1:opt.nit % bisection iterations

        % plot in debug mode
        if opt.debug&&(l==1)
          plot(gam_low,etatr(gam_low),'>')
          plot(gam_high,etatr(gam_high),'<')
          plot(gam,etatr(gam),'+')
        end

        % revise search parameters
        if etatr(gam) < 1+in(nz(l))
          gam_high = gam;
        else
          gam_low = gam;
        end
        gam = 0.5*(gam_low+gam_high);

        % stopping tolerance
        if abs(gam_low-gam)/gam_low < opt.tol, 
          break; 
        end

      end %t

      % check if stopping tolerance met
      if opt.verbose && (abs(gam_low-gam)/gam_low > opt.tol), 
        warning('tol not reached')
      end

      % compute s-transform
      out(l) = -gam*(1+in(nz(l)))/in(nz(l));

    end %l


  case 'newton'
    etatr_deriv = @(gam) -mean(bsxfun(@rdivide,eig,(1+eig*gam).^2),1);
    
    for l=1:length(nz) % for each non-zero input ...

      % compute inverse eta-transform
      gam = (1/(1+in(nz(l))) -1)/eig_mean; % initialization
      for t=1:opt.nit

        % plot in debug mode
        if opt.debug
          plot(gam,etatr(gam),'+')
        end

        % newton iteration
        gam_old = gam;
        gam = gam - (etatr(gam)-1-in(nz(l)))/etatr_deriv(gam);

        % stopping tolerance
        if abs(gam_old-gam)/gam_old < opt.tol, 
          break; 
        end

      end %t

      % check if stopping tolerance met
      if opt.verbose && (abs(gam_old-gam)/gam_old > opt.tol), 
        warning('tol not reached')
      end

      % compute s-transform
      out(l) = -gam*(1+in(nz(l)))/in(nz(l));

    end %l

  otherwise
    error('3rd input must be either ''newton'' or ''bisection''.')

end % opt.type

if opt.debug
  hold off
end
