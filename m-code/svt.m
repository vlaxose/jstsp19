function X = svt(Y, tau)

  [Mr, Mt] = size(Y);
  
  [Uy,Sy,Vy] = svd(Y);
  lambda = diag(Sy);
  softThres = diag(max(0, lambda-tau).*lambda./abs(lambda));
  if(~isnan(softThres))
    SS = [softThres zeros(size(softThres, 1), size(Vy, 1)-size(softThres,1)) ; zeros(size(Uy, 1)-size(softThres,1), size(softThres, 1)) zeros(size(Uy, 1)-size(softThres,1), size(Vy, 1)-size(softThres,1))];
    X = Uy*SS*Vy';
  else
    X = zeros(Mr, Mt);
  end

end