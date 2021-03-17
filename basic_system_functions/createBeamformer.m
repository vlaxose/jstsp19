function B = createBeamformer(N, beamformer_type)

% Create the beamformers
   switch(beamformer_type)
    case 'fft'
     B = 1/sqrt(N)*fft(eye(N));
    case 'rand'
     B = 1/sqrt(N)*randsrc(N,N, [1 -1 1j -1j]);
    case 'rand_ps'
     Gr = 32;
     B = 1/sqrt(N)*exp(-1j*(0:N-1)'*2*pi*randi(Gr, 1, N)/Gr);
    case 'ps'
     Gr = N;
     B = 1/sqrt(N)*exp(-1j*(0:N-1)'*2*pi*[0:Gr-1]/Gr);
    case 'ZC' % Zadoff-Chu
     R = 11;
     B = 1/sqrt(N)*exp(-1j*R*(0:N-1)'*pi*(1:N)/N);
    case 'quantized_4'
     N_q = 4;
     A = [0:2^N_q-1];
     K = ceil(N/length(A));
     A = vec(kron(ones(K,1), A)).';
     omega  = 2*pi/2^N_q *A(1:N);
     B = 1/sqrt(N)*exp(-1j*(0:N-1)'*omega);
    case 'quantized'
     N_q = 6;
     A = [0:2^N_q-1];
     K = ceil(N/length(A));
     A = vec(kron(ones(K,1), A)).';
     omega  = 2*pi/2^N_q *A(1:N);
     B = 1/sqrt(N)*exp(-1j*(0:N-1)'*omega);
   end
  
   
end