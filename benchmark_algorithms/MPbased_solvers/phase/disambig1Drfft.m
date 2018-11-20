% Finds the flipped, circ-shifted, and phase-rot version of "in" that best matches "ref"
% 
% [out,fxn]=disambig1Drfft(in,ref)

function [out,fxn]=disambig1Drfft(in,ref)

 % extract input size
 n=length(in);

 minErr = inf;
 for flip=0:1
   if flip,
     in_flip = flipud(in);
   else
     in_flip = in;
   end
   for kk=0:n-1
     in_shift = circshift(in_flip,kk);
     angl = sign(in_shift'*ref);
     in_rot = in_shift*angl;
     err = norm(in_rot-ref);
     if err<minErr
       minErr = err;
       out = in_rot;
       kk_best = kk;
       flip_best = flip;
       angl_best = angl;
     end
   end
 end

 if nargout>1
   if flip_best
     fxn = @(invec) circshift(flipud(invec),kk_best)*angl_best;
   else
     fxn = @(invec) circshift(invec,kk_best)*angl_best;
   end
 end
