function res = mtimes(a,b)
if a.adjoint
    % x=H'*y (x=res,b=y), x: object, y: multi-coil k-space data
    % multi-coil data in image domain 
    for ch=1:size(b,4),
        x_array(:,:,:,ch)=ifft2c_mri(b(:,:,:,ch).*a.mask);
    end
    % multi-coil combination in the image domain
    for tt=1:size(b,3),
        res(:,:,tt)=sum(squeeze(x_array(:,:,tt,:)).*conj(a.b1),3)./sum(abs((a.b1)).^2,3); %#ok<AGROW>
        %res(:,:,tt)=sum(squeeze(x_array(:,:,tt,:)).*conj(a.b1),3); %#ok<AGROW>
    end
else
    % y=H*x (y=res,b=x), x: object, y: multi-coil k-space data
    % multi-coil image from object
    for tt=1:size(b,3),
    for ch=1:size(a.b1,3),
        x_array(:,:,tt,ch)=b(:,:,tt).*a.b1(:,:,ch); %#ok<AGROW>
    end
    end
    % multi-coil image to k-space domain
    res=fft2c_mri(x_array);
    % apply sampling mask
    for ch=1:size(a.b1,3),
        res(:,:,:,ch)=res(:,:,:,ch).*a.mask;
    end
end