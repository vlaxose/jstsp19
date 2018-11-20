classdef ParametricZ < hgsetget
    %ParametricZ Class to compute several required quantities for use by
    %P-BiG-AMP and related codes. A class level interface for these
    %computations is being provided to allow for a variety of different
    %implementation strategies to be used for different problems classes.
    %Code deals with noiseless measurements z (M x 1) which are a
    %deterministic bilinear function of
    %parameters b (Nb x 1) and c (Nc x 1)
    
    properties
        
        
    end
    
    methods (Abstract)
        
        [M,Nb,Nc] = returnSizes(obj)
        %Return all problem dimensions
        
        [z,pvarBar,pvar] = pComputation(obj,opt,bhat,nub,chat,nuc)
        %Method computes z(bhat,chat), pvar, and pvarBar based on the
        %P-BiG-AMP derivation given the specified inputs. opt is an object
        %of class PBiGAMPOpt
        
        [rhat,rvar,qhat,qvar] = rqComputation(obj,opt,bhat,nub,chat,nuc,shat,nus)
        %Method computes q and r based on the P-BiG-AMP derivation
        %given the specified inputs. opt is an object of class PBiGAMPOpt
        
        z = computeZ(obj,bhat,chat)
        %Compute z(bhat,chat)
        
    end
    
    methods
        
        
        
    end
    
end

