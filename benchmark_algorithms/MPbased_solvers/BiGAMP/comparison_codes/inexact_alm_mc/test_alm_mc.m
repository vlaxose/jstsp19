% This script is used to test the Inexact ALM algorithm for Matrix Completion on randomly generated matrices.
%
% Minming Chen, October 2009. Questions? v-minmch@microsoft.com ; 
% Arvind Ganesh (abalasu2@illinois.edu)
%
% Copyright: Perception and Decision Laboratory, University of Illinois, Urbana-Champaign
%            Microsoft Research Asia, Beijing

newDataFlag = 1;

if newDataFlag == 1
    
    clc ;
    close all ;
    m = 1000 ;
    pdr = 6;
    n = m ;
    r = 10;
    ML = (randn(m,r)); MR = (randn(n,r));
    p = min(round(r * (2 * n - r) * pdr), m*n);
    rho_s = p / (m * n);
    
    [I J col omega] = myRandsample(m, n, p);
    V = UVtOmega(ML, MR, I, J, col);
    
    D = spconvert([I,J,V; m,n,0]);
%     clear I J col;
    
end

disp('m');
disp(m);
disp('n');
disp(n);
disp('r');
disp(r);
disp('p');
disp(p);
disp('pdr');
disp(pdr);
disp('rho_s');
disp(rho_s);

% inexact alm mc
tic;
[A iter svp] = inexact_alm_mc(D, 1e-4);
tElapsed = toc;

trAA = sum(sum((A.U'*A.U) .* (A.V'*A.V)));
trMM = sum(sum((ML'*ML) .* (MR'*MR)));
trAM = sum(sum((A.U'*ML) .* (A.V'*MR)));
error = sqrt((trAA + trMM -2*trAM)/trMM);

disp('Iteration');
disp(iter);
disp('Time');
disp(tElapsed);
disp('rank of A');
disp(svp);
disp('|A-M|_F / |M|_F');
disp(error);

