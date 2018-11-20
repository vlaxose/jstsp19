function res = mtimes(a,b)

if a.adjoint
    % to image space: n-coil -> 1-coil
    if size(a.mask,3)>1,
        for ch=1:size(a.b1,3),
            aux(:,:,ch)=ifft2c_mri(b(:,:,ch).*a.mask(:,:,ch)); %#ok<AGROW>
        end
    else
        for ch=1:size(a.b1,3),
            aux(:,:,ch)=ifft2c_mri(b(:,:,ch).*a.mask); %#ok<AGROW>
        end
    end
    %res=sum(aux.*conj(a.b1),3);
    res=sum(aux.*conj(a.b1),3)./sum(abs((a.b1)).^2,3);
    res(isnan(res))=0;
else
    % to k-space: 1-coil -> n-coil
    if size(a.mask,3)>1,
        for ch=1:size(a.b1,3),
            res(:,:,ch)=fft2c_mri(b.*a.b1(:,:,ch)).*a.mask(:,:,ch); %#ok<AGROW>
        end
    else
        for ch=1:size(a.b1,3),
            res(:,:,ch)=fft2c_mri(b.*a.b1(:,:,ch)).*a.mask; %#ok<AGROW>
        end
    end
end





    
