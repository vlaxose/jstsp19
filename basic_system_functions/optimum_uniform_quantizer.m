
function [output, up, down] = optimum_uniform_quantizer(input, bits)

  if(~isinteger(int8(bits)))
    error('Wrong value for bits (2nd input argument)');
  elseif(bits>8)
    stepsize = 0.01;
  else
    optimum_stepsize = [ 1.5958 0.9957 0.586 0.3352 0.1881 0.1041 0.0569 0.0308];
    stepsize = optimum_stepsize(bits);
  end
  
  input_real = real(input);
  abs_input_real = abs(input_real);
  input_imag = imag(input);
  abs_input_imag = abs(input_imag);
  Dreal = sqrt(mean(abs_input_real.^2))*stepsize;
  Dimag = sqrt(mean(abs_input_imag.^2))*stepsize;
  output = sign(input_real).*(min(ceil(abs_input_real/Dreal), 2^(bits-1))-1/2)*Dreal...
           + 1j*sign(input_imag).*(min(ceil(abs_input_imag/Dimag), 2^(bits-1))-1/2)*Dimag;
  output_real = real(output);
  output_imag = imag(output);
  up = output_real + 1/2*Dreal +1j*(output_imag + 1/2*Dimag);
  down = output_real - 1/2*Dreal +1j*(output_imag - 1/2*Dimag);
  
end