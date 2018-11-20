% function to return v based on D and desired Bayes Error Rate using a
% bisectional search method

function v = BayesError2variance(BER,D)

% v is signal variance
% BER is desired Bayes Error Rate
% D is the number of categories

% set detection tolerance (abs(Pe_guess - Pe ) < Pe*tol)
tol = 1e-4;

npts = 1000; % number of integration points

if D==2
    v = (1/(sqrt(2)*norminv((1-BER),0,1)))^2;
else
    
    %calculate upper bound on v by using the exact for D=2
    v_u=(1/(sqrt(2)*norminv((1-BER),0,1)))^2;
    
    %calculate lower bound on v by using the Union Bound
    Ppwe = 1-(1-BER)^(1/(D-1));
    v_l=(1/(sqrt(2)*norminv((1-Ppwe),0,1)))^2;
    
    stop = 1;
    counter = 0;
    while stop
        counter = counter+1;
        
        v_guess = (v_u+v_l)/2;
        
        z = linspace(-6,6,npts) + 1/sqrt(v_guess);
        
        % integrand definition
        Pc=normpdf(z,1/sqrt(v_guess),1).*(normcdf(z,0,1).^(D-1));
        
        % integration
        Pc=trapz(Pc)*(z(2)-z(1));
        
        Pe_guess=1-Pc;
        
        % check for convergence
        if abs(Pe_guess-BER)<BER*tol || counter == 100
            stop = 0;
            v=v_guess;
        % check for which region
        elseif Pe_guess > BER
            v_u = v_guess;
        else
            v_l = v_guess;
        end
            
    end
   
end

end

