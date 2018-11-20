function elem = getElement(E)
elem = cell(3,1);
elem{1} = E.adjoint;
elem{2} = E.mask;
elem{3} = E.b1;
end
