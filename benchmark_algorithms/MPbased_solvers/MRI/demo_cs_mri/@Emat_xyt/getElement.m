function elem = getElement(E_xyt)
elem = cell(3,1);
elem{1} = E_xyt.adjoint;
elem{2} = E_xyt.mask;
elem{3} = E_xyt.b1;
end
