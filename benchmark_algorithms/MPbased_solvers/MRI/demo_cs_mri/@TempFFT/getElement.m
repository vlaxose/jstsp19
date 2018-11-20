function elem = getElement(TempFT)
elem = cell(2,1);
elem{1} = TempFT.adjoint;
elem{2}= TempFT.dim;
end
