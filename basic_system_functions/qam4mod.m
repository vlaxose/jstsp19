function [symbols] = qam4mod(input, mode, N)
  
  switch(mode)
    
    % 4-QAM modulation
    case 'mod'
      alphabet = [(1+1j)/sqrt(2) (-1+1j)/sqrt(2) (1-1j)/sqrt(2) (-1-1j)/sqrt(2)];
      symbols = randsrc(N, 1, alphabet);

      
    % 4-QAM demodulation
    case 'demod'
      softDecision = input;

      [N1, N2] = size(softDecision);
      onesMatrix = ones(N1, N2);
      conjSymbols = -1j*softDecision;

      % Normalized real and imaginary softDecision
      s_real = 1 / sqrt(2);
      s_imag = 1j * s_real;

      % Find the symbol indexes for each one of 4 softDecision of 4qam modulation.
      s2_indx = (softDecision >= 0 & conjSymbols <=0);
      s3_indx = (softDecision <= 0 & conjSymbols >=0);
      s4_indx = (softDecision <= 0 & conjSymbols <=0);

      % Quantize the symbol matrix
      symbols = (s_real + s_imag) * onesMatrix;
      symbols(s2_indx) = s_real - s_imag;
      symbols(s3_indx) = - s_real + s_imag;
      symbols(s4_indx) = - s_real - s_imag;
  end
end