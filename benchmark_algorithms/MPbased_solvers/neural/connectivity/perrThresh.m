function [perr, topt, pfa, pmd] = perrThresh(xtot, xhattot)

nbeta = size(xhattot,2);
perr = zeros(nbeta,1);
topt = zeros(nbeta,1);
nthresh = 500;
pfa = zeros(nthresh,nbeta);
pmd = zeros(nthresh,nbeta);
for ibeta = 1:nbeta
    xmax = max(abs(xtot(:,ibeta)));
    thtest = [0; logspace(log10(xmax)-6,log10(xmax),nthresh)'];
    nth = length(thtest);
    J = (abs(xtot(:,ibeta)) > 1e-3);
    xhatabs = abs(xhattot(:,ibeta));
    perrT = zeros(nth,1);
    for ith = 1:nth
        Jhat = (xhatabs >= thtest(ith));
        perrT(ith) = mean(J ~= Jhat);
        pfa(ith,ibeta) = sum(~J & Jhat) / sum(~J);
        pmd(ith,ibeta) = sum(J & ~Jhat) / sum(J);
    end
    [mm,im] = min(perrT);
    perr(ibeta) = mm;
    topt(ibeta) = thtest(im);
    
end
if 0
plot(thtest,perrT);
grid on;
xx = 1;
end
