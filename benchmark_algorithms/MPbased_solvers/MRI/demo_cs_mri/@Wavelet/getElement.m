function elem = getElement(W)
elem = cell(3,1);
elem{1} = W.adjoint;
elem{2}=W.qmf;
elem{3}=W.wavScale;
end
