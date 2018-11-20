%A simple program to test ToeplitzLinTrans
clear all
clc

% problem parameters
Lrow=500;		% length of first row in toeplitz mtx
Lcol=200;	% length of first column in toeplitz mtx
Dcol=2;		% column decimation factor
Drow=3;		% row decimation factor
omit = [1,2,5];	% post-decimation row-puncturing indices

% create inputs
Lin_fwd = ceil(Lrow/Dcol);
in_fwd = randn(Lin_fwd,2)*[1;1j];
Lin_bak = ceil(Lcol/Drow)-length(omit);
in_bak = randn(Lin_bak,2)*[1;1j];
first_col = randn(Lcol,2)*[1;1j];
first_row = [first_col(1);randn(Lrow-1,2)*[1;1j]];

%Build the A operator
whichR = 1:Drow:Lcol; %downsample
whichC = 1:Dcol:Lrow; %downsample
whichR = whichR(setdiff(1:length(whichR),omit));
Aop = ToeplitzLinTrans(first_row,first_col,whichR,whichC);

%Try forward matrix mult
out_fwd = Aop.mult(in_fwd);
out_fwd2 = Aop.multSq(in_fwd);
out_bak = Aop.multTr(in_bak);
out_bak2 = Aop.multSqTr(in_bak);


% create downsampled row-punctured toeplitz matrix for testing
A_toep = toeplitz(first_col,first_row);
A_down = A_toep(1:Drow:end,1:Dcol:end);
A = A_down(setdiff(1:size(A_down,1),omit),:);

% explicit matrix multiplication for testing
out_fwd_test = A*in_fwd;
out_fwd2_test = (abs(A).^2)*in_fwd;
out_bak_test = A'*in_bak;
out_bak2_test = (abs(A).^2)'*in_bak;


% errors
max(abs(out_fwd-out_fwd_test))
max(abs(out_fwd2-out_fwd2_test))
max(abs(out_bak-out_bak_test))
max(abs(out_bak2-out_bak2_test))

