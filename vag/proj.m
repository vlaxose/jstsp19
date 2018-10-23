function B = proj(A)

    [M,N] = size(A);
    B = zeros(M,N);
    for m = 1:M
        for n = 1:N
            if(A(m,n)>=1)
                B(m,n) = 1;
            elseif( 1e-7<A(m,n) && A(m,n)<1 )
                B(m,n) = A(m,n);
            else
                B(m,n) = 0;
            end
        end
    end

end