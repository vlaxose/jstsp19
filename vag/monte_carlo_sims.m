function [mean_rate, mean_ee, mean_Lt_opt, mean_kappa] = monte_carlo_sims(transmit_snr, Nt, Nr, Lt, Ns, K, maxDBiters)

  total_num_of_clusters = 2;
  total_num_of_rays = 10;
  
  total_monte_carlo_realizations = 1;
  converged_monte_carlo_realizations  = 0;
  mean_rate = zeros(1, K);
  mean_ee = zeros(1, K);
  mean_Lt_opt = zeros(1, K);
  mean_kappa = zeros(1, maxDBiters+1);
  
  parfor r=1:total_monte_carlo_realizations

    [converged, rate, ee, Lt_opt, ~, kappa] = systemModel(Nt, Nr, Lt, Ns, total_num_of_clusters, total_num_of_rays, transmit_snr, maxDBiters, 0);

    if(converged==1)
      mean_rate = mean_rate+rate;
      mean_ee = mean_ee + ee;
      mean_Lt_opt = mean_Lt_opt + Lt_opt;
      mean_kappa = mean_kappa + kappa;
      converged_monte_carlo_realizations = converged_monte_carlo_realizations + 1;
    end

  end
  
  if(converged_monte_carlo_realizations>0)
   mean_rate = mean_rate/converged_monte_carlo_realizations;
   mean_ee = mean_ee/converged_monte_carlo_realizations;
   mean_Lt_opt = mean_Lt_opt/converged_monte_carlo_realizations;
   mean_kappa = mean_kappa/converged_monte_carlo_realizations;
  end
  converged_monte_carlo_realizations
  
end
