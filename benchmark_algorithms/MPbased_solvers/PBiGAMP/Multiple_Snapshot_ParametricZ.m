classdef Multiple_Snapshot_ParametricZ < ParametricZ
    %This class implements the ParametricZ functionality for a model of the
    %form
    %Z = A(b)*C
    %where A(b) is KxN, C is NxL, and Z is KxL.
    %This yields M = K*L for the measurements z = vec(Z).
    %Notice that the parameters are the vector b and c = vec(C),
    %which yields Nc = N*L
    
    
    
    properties
        
        %The PLinTrans operator
        A;
        
        %The number of snapshots
        L;
        
        %Size of the operator. Copied from A by constructor
        K;N;
        
        %Useful z values
        zij_sumSquared;
        zij_sumMi;
        zij_sumMj;
    end
    
    
    methods
        
        %% Default constructor
        function obj = Multiple_Snapshot_ParametricZ(A,L)
            
            %Verify that A is a PLinTrans object
            if ~isa(A,'PLinTrans')
                error('A must be a PLinTrans object')
            end
            
            %Verify that L is scalar
            if numel(L) > 1
                error('L should be scalar')
            end
            
            %Assign
            obj.A = A;
            obj.L = L;
            
            %Get sizes
            [obj.K,obj.N] = A.getSizes();
            
            
            %Pre-compute
            [AiNorms,AnNorms] = obj.A.computeAFrobNorms;
            obj.zij_sumMj = obj.L*AiNorms.^2;
            obj.zij_sumMi = repmat(AnNorms.^2,obj.L,1);
            obj.zij_sumSquared = sum(obj.zij_sumMj);
        end
        
        
        %% returnSizes
        %Return all problem dimensions
        function [M,Nb,Nc] = returnSizes(obj)
            
            %Obtain sizes
            M = obj.L*obj.K;
            Nb = obj.A.parameterDimension;
            Nc = obj.N*obj.L;
            
        end
        
        
        %% computeZ
        %Evalute the model for a given bhat and chat.
        function Z = computeZ(obj,bhat,chat)
            
            
            %Set the parameter
            obj.A.setParameter(bhat);
            
            %Do the multiplication
            Z = vec(obj.A.mult(reshape(chat,obj.N,obj.L)));
            
            
        end
        
        
        %% pComputation
        %Method computes z(bhat,chat), pvar, and pvarBar based on the
        %P-BiG-AMP derivation given the specified inputs. opt is an object
        %of class PBiGAMPOpt
        function [z,pvarBar,pvar] = pComputation(obj,opt,bhat,nub,chat,nuc)
            
            %First, set parameter
            obj.A.setParameter(bhat);
            
            
            
            %Computations depend on uniformVariance flag
            if ~opt.uniformVariance
                
                %Reshape
                chat = reshape(chat,obj.N,obj.L);
                nuc = reshape(nuc,obj.N,obj.L);
                
                %First, get Zi0
                Zi0 = obj.A.multWithDerivatives(chat);
                
                %Get the squared derivative multiplies
                Zi2 = obj.A.multWithSqrDerivatives(nuc);
                
                %Compute pvarBar
                pvarBar = vec(obj.A.multSq(nuc)) + (abs(Zi0).^2)*nub;
                
                %Compute pvar
                pvar = pvarBar + Zi2*nub;
                
                %Compute z
                z = vec(obj.A.mult(chat));
                
            else
                
                %Get M
                M = obj.returnSizes();
                
                %Compute z
                z = obj.computeZ(bhat,chat);
                
                %Reshape C
                chat = reshape(chat,obj.N,obj.L);
                
                %Get sums
                sumZi0 = obj.A.computeFrobSum(chat);
                
                %Compute the sum over Z0j
                sumZ0j = obj.L*obj.A.computeFrobNorm()^2;
                
                %Compute pvarBar
                pvarBar = (nub*sumZi0 + nuc*sumZ0j)/M;
                
                %Compute pvar
                pvar = pvarBar + nub*nuc/M*obj.zij_sumSquared;
                
            end
            
        end
        
        
        %% rqComputation
        
        %Method computes Q and R based on the P-BiG-AMP derivation
        %given the specified inputs. opt is an object of class PBiGAMPOpt
        function [rhat,rvar,qhat,qvar] = rqComputation(...
                obj,opt,bhat,nub,chat,nuc,shat,nus)
            
            %First, set parameter
            obj.A.setParameter(bhat);
            
            %Get sizes
            [~,Nb,Nc] = obj.returnSizes();
            
            %Handle uniform variance
            if ~opt.uniformVariance
                
                %Reshape
                chat = reshape(chat,obj.N,obj.L);
                nuc = reshape(nuc,obj.N,obj.L);
                
                %First, get Zi0- go ahead and conjugate transpose
                Zi0 = (obj.A.multWithDerivatives(chat))';
                
                %Compute qhat single sum term
                qhat = Zi0*shat;
                
                %Compute qvar (currently inverted)
                qvar = abs(Zi0).^2*nus;
                
                
                %Reshape s terms
                shat = reshape(shat,obj.K,obj.L);
                nus = reshape(nus,obj.K,obj.L);
                
                %Compute rhat single sum term
                rhat = vec(obj.A.multTr(shat));
                
                %Directly compute rvar (currently inverted)
                rvar = vec(obj.A.multSqTr(nus));
                
                %Revectorize chat
                chat = vec(chat);
                
                %Handle double sum terms
                rhat = rhat - chat .* ...
                    vec(obj.A.computeWeightedSum(nub)'*nus);
                
                %qhat double sum term
                Zi2 = obj.A.multWithSqrDerivatives(nuc);
                qhat = qhat - bhat .* vec(Zi2'*vec(nus));
                
            else
                
                %Reshape chat
                chat = reshape(chat,obj.N,obj.L);
                
                %First, get Zi0- go ahead and conjugate transpose
                Zi0 = (obj.A.multWithDerivatives(chat))';
                
                %Compute qhat single sum term
                qhat = Zi0*shat;
                
                %Reshape shat term
                shat = reshape(shat,obj.K,obj.L);
                
                %Compute rhat single sum term
                rhat = vec(obj.A.multTr(shat));
                
                %Get sums
                sumZi0 = obj.A.computeFrobSum(chat);
                
                %Compute the sum over Z0j
                sumZ0j = obj.L*obj.A.computeFrobNorm()^2;
                
                %Compute rvar (currently inverted)
                rvar = nus/Nc*sumZ0j;
                
                %Compute qvar (currently inverted)
                qvar = nus/Nb*sumZi0;
                
                %Revectorize chat
                chat = vec(chat);
                
                %Double sum term
                rhat = rhat - nus*nub*(chat .* obj.zij_sumMi);
                qhat = qhat - nus*nuc*(bhat .* obj.zij_sumMj);
                
            end
            
            %Invert the variance computations
            rvar = 1 ./ (rvar + realmin);
            qvar = 1 ./ (qvar + realmin);
            
            %Enforce variance limits
            rvar(rvar > opt.varThresh) = opt.varThresh;
            qvar(qvar > opt.varThresh) = opt.varThresh;
            
            
            %Scale the rhat and qhat terms by the variances and then add in
            %the chat and bhat terms
            rhat = rhat .* rvar;
            rhat = rhat + chat;
            qhat = qhat .* qvar;
            qhat = qhat + bhat;
            
        end
        
    end
    
    
    
end
