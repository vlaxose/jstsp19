disp('Comparing MultiSNIPEstim against two SNIPEstim (translated to zero if nec.) ');
disp('The expectation is that the MultiSNIPEstim output quantities')
disp('will locally match one of the SNIPEstim in all respects')
centers=[0 1];
omega=[5 2];
rhat = linspace(-.5,2,250)';
rvar = 1e-2;

% compare against single-delta SNIPE and non-zero offset SNIPE
s0.e = SNIPEstim(omega(1));
[s0.xhat,s0.xvar,s0.val] = s0.e.estim(rhat,rvar);
s0.logScale = s0.e.logScale( s0.xhat,rvar,rhat);
s1.e = SNIPEstim(omega(2));
rhatOffset = rhat-centers(2);
[s1.xhat,s1.xvar,s1.val] = s1.e.estim(rhatOffset,rvar);
s1.xhat = s1.xhat + centers(2);

s1.logScale = s1.e.logScale( s1.xhat-centers(2),rvar,rhatOffset);

% compare to MultiSNIPEstim 
m = MultiSNIPEstim(centers,omega);
[xhat,xvar,val] = m.estim(rhat,rvar);
logScale = m.logScale( xhat,rvar,rhat);

disp('with the exception of a constant offset for the logScale and value quantities')
% The log-likelihoods contain constant offsets that do not affect maximization
s0.logScale = s0.logScale - min(s0.logScale); s1.logScale = s1.logScale - min(s1.logScale); logScale = logScale - min(logScale);
s0.val = s0.val - min(s0.val); s1.val = s1.val - min(s1.val); val = val - min(val);

quants = {'xhat','xvar','logScale','val'};
titles.xhat = 'posterior estimate (xhat)';
titles.xvar = 'posterior variance (xvar)';
titles.logScale = 'log(int pRX pX )+Hgauss (.logScale method)';
titles.val = '-KL Divergence ("value" output from estim)';

for k=1:4
    figure(k)
    qk = quants{k}; % e.g. 'xhat'
    plot(rhat,eval(qk),'o')
    hold all
    plot(rhat,[s0.(qk) s1.(qk) ])
    hold off
    xlabel('rhat')
    ylabel(qk);
    legend('MultiSNIPEstim', 'centered at 0','centered at 1')
    grid minor
    title(['Comparing ' titles.(qk)] )
end
