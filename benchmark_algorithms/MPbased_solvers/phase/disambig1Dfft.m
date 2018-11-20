% Finds the flipped, conjugated, circ-shifted, and phase-rot version of "in" that best matches "ref"
% 
% [out,fxn]=disambig1Dfft(in,ref)

function [out,fxn]=disambig1Dfft(in,ref)

 % extract input size
 n=length(in);

 minErr = inf;
 for flip=0:1
   if flip,
     in_flip = flipud(in);
   else
     in_flip = in;
   end
   for con=0:1
     if con,
       in_con = conj(in_flip);
     else
       in_con = in_flip;
     end
     for kk=0:n-1
       in_shift = circshift(in_con,kk);
       angl = sign(in_shift'*ref);
       in_rot = in_shift*angl;
       err = norm(in_rot-ref);
       if err<minErr
         minErr = err;
         out = in_rot;
         kk_best = kk;
         flip_best = flip;
         con_best = con;
         angl_best = angl;
       end
     end
   end
 end

 if nargout>1
   if flip_best
     if con_best
       fxn = @(invec) circshift(conj(flipud(invec)),kk_best)*angl_best;
     else
       fxn = @(invec) circshift(flipud(invec),kk_best)*angl_best;
     end
   else
     if con_best
       fxn = @(invec) circshift(conj(invec),kk_best)*angl_best;
     else
       fxn = @(invec) circshift(invec,kk_best)*angl_best;
     end
   end
 end
