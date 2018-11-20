% test concatenation of estimators and linear transforms
clear all;

nx_top =  200; 
nx_bot =  300; 
nx = nx_top + nx_bot;
nz_top = 250;
nz_bot = 350;
nz = nz_top+nz_bot;

E_x = randn(nx,1);
V_x = rand(nx,1)*.16;
V_y = rand(nz,1)*.16;

A = (1/sqrt(nx))*randn(nz,nx);
x=randn(nx,1).*sqrt(V_x)+E_x;
y=A*x+randn(nz,1).*sqrt(V_y);

% LMMSE baseline
fprintf(1,'Calculating baseline LMMSE\n');
xhatLMMSE = E_x + V_x.*( A'*((A*diag(V_x)*A' + diag(V_y) ) \ ( y - A*E_x)));
MSE_LMMSE = 20*log10(norm(xhatLMMSE-x)/norm(x));
if MSE_LMMSE > -12
    error 'LMMSE did not find a good solution'
end

%GAMP: full in, full out 
fprintf(1,'Calculating GAMP with full vectors\n');
xhatGamp1 = gampEst(AwgnEstimIn( E_x,V_x), AwgnEstimOut(y, V_y, 0) ,A );
MSE_Gamp1 = 20*log10(norm(xhatGamp1-x)/norm(x));
if MSE_Gamp1 > MSE_LMMSE+.1
    error 'gampEst solution did not agree with LMMSE'
end

%GAMP: concatenated input, full output 
fprintf(1,'Calculating GAMP with concatenated input\n');
estimInArray = cell(2,1);
estimInArray{1} = AwgnEstimIn( E_x(1:nx_top),V_x(1:nx_top) );
estimInArray{2} = AwgnEstimIn( E_x(nx_top+1:end),V_x(nx_top+1:end) );
eiConc = EstimInConcat( estimInArray, [nx_top nx_bot]');
xhatGamp2 = gampEst( eiConc, AwgnEstimOut(y, V_y, 0) ,A );
MSE_Gamp2 = 20*log10(norm(xhatGamp2-x)/norm(x));
if MSE_Gamp2 > MSE_LMMSE+.1
        error 'gampEst solution (EstimInConcat) did not agree with LMMSE'
end 

%GAMP: concatenated input,concatenated output
fprintf(1,'Calculating GAMP with concatenated input,concatenated output\n');
estimOutArray = cell(2,1);
estimOutArray{1} = AwgnEstimOut(y(1:nz_top), V_y(1:nz_top), 0);
estimOutArray{2} = AwgnEstimOut(y(1+nz_top:end), V_y(1+nz_top:end), 0);
eoConc = EstimOutConcat( estimOutArray, [nz_top nz_bot]');
xhatGamp3 = gampEst(eiConc, eoConc ,A );
MSE_Gamp3 = 20*log10(norm(xhatGamp2-x)/norm(x));
if MSE_Gamp3 > MSE_LMMSE+.1
    error 'gampEst solution (EstimInConcat+EstimOutConcat) did not agree with LMMSE'
end 

%GAMP: horizontally concatenated LinTrans
ltc =  LinTransConcat({MatrixLinTrans(A(:,1:nx_top)) MatrixLinTrans(A(:,1+nx_top:end)) });
xhatGamp4 = gampEst(AwgnEstimIn( E_x,V_x), AwgnEstimOut(y, V_y, 0) ,ltc );
MSE_Gamp4 = 20*log10(norm(xhatGamp2-x)/norm(x));
if MSE_Gamp4 > MSE_LMMSE+.1
    error 'gampEst solution (LinTransConcat) did not agree with LMMSE'
end

%GAMP: vertically concatenated LinTrans
ltc =  LinTransConcat({A(1:nz_top,:); A(1+nz_top:end,:) });
xhatGamp5 = gampEst(AwgnEstimIn( E_x,V_x), AwgnEstimOut(y, V_y, 0) ,ltc );
MSE_Gamp5 = 20*log10(norm(xhatGamp2-x)/norm(x));
if MSE_Gamp5 > MSE_LMMSE+.1
    error 'gampEst solution (LinTransConcat) did not agree with LMMSE'
end


fprintf(1,'EstimInConcat, EstimOutConcat,LinTransConcat tests passed\n');
