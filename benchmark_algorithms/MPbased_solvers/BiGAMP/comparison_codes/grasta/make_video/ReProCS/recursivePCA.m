
function[Ut Sigt Pdel]=recursivePCA(Lt,U0,Sig0)

global D d tau sig_add sig_del
D = [D Lt];d = d+1;  
if ( d < tau)
    Ut=U0; Sigt=Sig0;
    Pdel=[];
else   
    %delete decayed directions
    a = (1/d)*diag((U0'*D)*(D'*U0),0); 
    T_del = find( abs(a) < sig_del); %min(a)
    Pdel = U0(:,T_del);  
    T_del_c = setdiff(1:size(U0,2),T_del);
    U0 = U0(:,T_del_c); Sig0 = Sig0(T_del_c,T_del_c);
    %incremental SVD
    L = U0'*D; H = D - U0*L;
    [J K] = qr(H,0);
    Sig = [Sig0,                  L ; 
           zeros(d,size(Sig0,2)), K ]; 
    [Ur Sigr ~] = svd(Sig,0);    
    Tr = union([1:size(U0,2)],find(abs(diag(Sigr))> sqrt(sig_add*d)));
    Ur = Ur(:,Tr); Sigr = Sigr(Tr,Tr);    
    Ut = [U0 J]*Ur;  Sigt = Sigr;  
    D = []; d = 0;  
end

