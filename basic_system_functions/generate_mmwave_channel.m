function [H,Ar,At,C] = generate_mmwave_channel(Mr, Mt, total_num_of_clusters, total_num_of_rays)
% Implementation of A. Forenza, D. J. Love and R. W. Heath, 
% "Simplified Spatial Correlation Models for Clustered MIMO Channels With Different Array Configurations,"
% in IEEE Transactions on Vehicular Technology, vol. 56, no. 4, pp. 1924-1934, July 2007.
% doi: 10.1109/TVT.2007.897212
% 
%
% Mr: the number of the receiving antennas
% Mt: the number of the transmiting antennas
%
% H: the channel matrix in angular domain
% Ar, At: the receiving and transmitting streering (phases) matrices
% C: the diagonal matrix with the channel gains
 
% Assumptions:
% - all the clusters have equal power (sigma_phi)
 
% Initialization
L = total_num_of_clusters*total_num_of_rays;
H = zeros(Mr, Mt);
C = zeros(L, L);
Hl = zeros(Mr, Mt);
Ar = zeros(Mr, L);
At = zeros(Mt, L);
Ghz = 90;
     
% Main
index = 1;
for tap = 1:total_num_of_clusters
   
    for ray=1:total_num_of_rays
        rayleigh_coeff = 1/sqrt(2)*(randn(1)+1j*randn(1));
        Ar(:, index) = angle(genLaplacianSamples(1), Mr);
        At(:, index) = angle(genLaplacianSamples(1), Mt);
         
        C(index, index) = rayleigh_coeff;
        Hl = Hl + rayleigh_coeff*Ar(:, index)*At(:, index)';
 
        index = index + 1;
    end
    H = H + Hl;
end
 
H = 1/sqrt(total_num_of_rays*total_num_of_clusters)*H;
end
 
% Generate the transmit and receive array responces
function vectors_of_angles=angle(phi, M)
    Ghz = 30;
    % For Uniform Linear Arrays (ULA) compute the phase shift
    wavelength = 30/Ghz; % w=c/lambda
    array_element_spacing = 0.5*wavelength;
    wavenumber = 2*pi/wavelength; % k = 2pi/lambda
    phi0 = 0; % mean AOA
    phase_shift = wavenumber*array_element_spacing*sin(phi0-phi)*(0:M-1).';
    vectors_of_angles = exp(1j*phase_shift);
end
 
 
% Random variable generator based on inverse transform sampling
function x=genLaplacianSamples(N)
    u = rand(N,1);
    % Inverse transform of trancuted Laplacian
    sigma_phi = 50; % standard deviation of the power azimuth spectrum (PAS)
    beta = 1/(1-exp(-sqrt(2)*pi/sigma_phi));
    x = beta*(exp(-sqrt(2)/sigma_phi*pi) - cosh(u));
end
