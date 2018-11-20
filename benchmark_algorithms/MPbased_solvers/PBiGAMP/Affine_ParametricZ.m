classdef Affine_ParametricZ < ParametricZ
    %This class implements the ParametricZ functionality for an arbitrary
    %bilinear form of parameters b and c. The Sandia National Labs
    %Tensor Toolbox is used for efficient computation to avoid the introduction
    %of for-loops.
    %http://www.sandia.gov/~tgkolda/TensorToolbox
    %The data model is given in terms of two paramter bectors b (Nb x 1) and
    %c (Nc x 1) with measurements Z (N x L) given by:
    %Z(m) =       \sum_{i,j} b(i) zij(m,i,j) c(j)
    %           + \sum_{i} zi0(m,i) b(i)
    %           + \sum_{j} z0j(m,j) c(j)
    %           + z00(m)
    %
    %The user provides the (possibly sparse) tensors:
    %zij (M x Nb x Nc)
    %zi0 (M x Nb)
    %z0j (M x Nc)
    %z00 (M)
    
    properties
        
        %(M x Nb x Nc) tensor
        zij;
        zij2; %Element-wise squared version. Stored for convenience.
        zijconj; %Element-wise conjugate. Stored for convenience.
        
        %(M x Nb) tensor. Leave empty to omit
        zi0;
        
        %(M x Nc) tensor. Leave empty to omit
        z0j;
        
        %(M) tensor. Leave empty to omit
        z00;
        
        %Quantities used by uniform variance
        zij_sumSquared; %sum of all squared entries
        zij_sumMi; %sum of squared entries over m and i (Nc x 1)
        zij_sumMj %sum of squared entries over m and j (Nb x 1)
    end
    
    
    
    methods
        
        %% Default Constructor
        function obj = Affine_ParametricZ(zij,zi0,z0j,z00)
            
            if nargin < 4
                z00 = [];
            end
            if nargin < 3
                z0j = [];
            end
            if nargin < 2
                zi0 = [];
            end
            if nargin == 0
                error('User must supply zij')
            end
            
            %Assign
            if ~isa(zij,'sptensor')
                obj.zij = zij;
                obj.zij2 = tenfun(@(q) abs(q).^2,zij);
                obj.zijconj = tenfun(@conj,zij);
            else
                %Set zij
                obj.zij = zij;
                obj.zij2 = elemfun(zij,@(q) abs(q).^2);
                obj.zijconj = elemfun(zij,@conj);
                
            end
            
            %Assign
            obj.zi0 = zi0;
            obj.z0j = z0j;
            obj.z00 = z00;
            
            %Uniform variance useful quantities
            obj.zij_sumSquared = collapse(obj.zij2);
            obj.zij_sumMi = collapse(obj.zij2,[1 2]);
            obj.zij_sumMj = collapse(obj.zij2,[1 3]);
        end
        
        
        
        %% returnSizes
        %Return all problem dimensions
        function [M,Nb,Nc] = returnSizes(obj)
            
            %Obtain sizes
            res = size(obj.zij);
            M = res(1);
            Nb = res(2);
            Nc = res(3);
            
        end
        
        
        %% computeZ
        %Evalute the model for a given bhat and chat.
        function Z = computeZ(obj,bhat,chat)
            
            %zij component
            Z = ttv(ttv(obj.zij,chat,3),bhat,2);
            
            %zi0 component
            if ~isempty(obj.zi0)
                Z = Z + ttv(obj.zi0,bhat,2);
            end
            
            %z0j component
            if ~isempty(obj.z0j)
                Z = Z + ttv(obj.z0j,chat,2);
            end
            
            %z00 component
            if ~isempty(obj.z00)
                Z = Z + obj.z00;
            end
            
            %Convert to matrix
            Z = double(Z);
        end
        
        
        %% pComputation
        %Method computes Z(bhat,chat), pvar, and pvarBar based on the
        %P-BiG-AMP derivation given the specified inputs. opt is an object
        %of class PBiGAMPOpt
        function [Z,pvarBar,pvar] = pComputation(obj,opt,bhat,nub,chat,nuc)
            
            %Get sizes
            M = obj.returnSizes();
            
            %Create zi*
            zis = ttv(obj.zij,chat,3);
            if ~isempty(obj.zi0)
                zis = zis + obj.zi0;
            end
            
            %Create z*j
            zsj = ttv(obj.zij,bhat,2);
            if ~isempty(obj.z0j)
                zsj = zsj + obj.z0j;
            end
            
            %Compute Z
            Z = ttv(zis,bhat,2);
            if ~isempty(obj.z0j)
                Z = Z + ttv(obj.z0j,chat,2);
            end
            if ~isempty(obj.z00)
                Z = Z + obj.z00;
            end
            
            %Compute elementwise squares
            if ~isa(zis,'sptensor')
                zis = tenfun(@(q) abs(q).^2,zis);
            else
                zis = elemfun(zis,@(q) abs(q).^2);
            end
            if ~isa(zsj,'sptensor')
                zsj = tenfun(@(q) abs(q).^2,zsj);
            else
                zsj = elemfun(zsj,@(q) abs(q).^2);
            end
            
            %Uniform Variance
            if ~opt.uniformVariance
                
                %Get pvarBar
                pvarBar = ttv(zis,nub,2) + ttv(zsj,nuc,2);
                
                %Compute pvar
                pvar = ttv(ttv(obj.zij2,nuc,3),nub,2) + pvarBar;
                
            else
                
                %Compute pvarBar
                pvarBar = (nub*collapse(zis) + nuc*collapse(zsj))/M;
                
                %Compute pvar
                pvar = pvarBar + nub*nuc/M*obj.zij_sumSquared;
                
            end
            
            %Convert to double
            Z = double(Z);
            pvarBar = double(pvarBar);
            pvar = double(pvar);
            
        end
        
        
        
        %% rqComputation
        
        %Method computes Q and R based on the P-BiG-AMP derivation
        %given the specified inputs. opt is an object of class PBiGAMPOpt
        function [rhat,rvar,qhat,qvar] = rqComputation(...
                obj,opt,bhat,nub,chat,nuc,shat,nus)
            
            %Get sizes
            [~,Nb,Nc] = obj.returnSizes();
            
            %Create zi*
            zis = ttv(obj.zij,chat,3);
            if ~isempty(obj.zi0)
                zis = zis + obj.zi0;
            end
            
            %Create z*j
            zsj = ttv(obj.zij,bhat,2);
            if ~isempty(obj.z0j)
                zsj = zsj + obj.z0j;
            end
            
            %Compute single sum term in rhat
            if ~isa(zsj,'sptensor')
                rhat = ttv(tenfun(@conj,zsj),shat,1);
            else
                rhat = ttv(tensor(elemfun(zsj,@conj)),shat,1);
            end
            
            %Compute single sum term in qhat
            if ~isa(zis,'sptensor')
                qhat = ttv(tenfun(@conj,zis),shat,1);
            else
                qhat = ttv(tensor(elemfun(zis,@conj)),shat,1);
            end
            
            
            %Uniform variance
            if ~opt.uniformVariance
                
                %Handle the double sum terms
                rhat = rhat + ...
                    chat.*ttv(tensor(ttv(obj.zij2,-1*nub,2)),nus,1);
                
                qhat = qhat + ...
                    bhat.*ttv(tensor(ttv(obj.zij2,-1*nuc,3)),nus,1);
                
                
                %Compute rvar, currently inverted
                if ~isa(zsj,'sptensor')
                    rvar = ttv(tenfun(@(q) abs(q).^2,zsj),nus,1);
                else
                    rvar = ttv(tensor(elemfun(zsj,@(q) abs(q).^2)),nus,1);
                end
                
                %Compute rvar, currently inverted
                if ~isa(zis,'sptensor')
                    qvar = ttv(tenfun(@(q) abs(q).^2,zis),nus,1);
                else
                    qvar = ttv(tensor(elemfun(zis,@(q) abs(q).^2)),nus,1);
                end
                
            else
                
                %Double sum term
                rhat = rhat - nus*nub*(chat .* obj.zij_sumMi);
                qhat = qhat - nus*nuc*(bhat .* obj.zij_sumMj);
                
                %Compute rvar, currently inverted
                if ~isa(zsj,'sptensor')
                    rvar = nus/Nc*collapse(tenfun(@(q) abs(q).^2,zsj));
                else
                    rvar = nus/Nc*collapse(elemfun(zsj,@(q) abs(q).^2));
                end
                
                %Compute rvar, currently inverted
                if ~isa(zis,'sptensor')
                    qvar = nus/Nb*collapse(tenfun(@(q) abs(q).^2,zis));
                else
                    qvar = nus/Nb*collapse(elemfun(zis,@(q) abs(q).^2));
                end
                
                
                
            end
            
            %Convert to double
            rhat = double(rhat);
            qhat = double(qhat);
            rvar = double(rvar);
            qvar = double(qvar);
            
            %Invert the variance computations
            rvar = 1 ./ (rvar + realmin);
            qvar = 1 ./ (qvar + realmin);
            
            %Enforce variance limits- notice that these are always dense
            %tensors due to the casts above
            rvar = min(rvar,opt.varThresh);
            qvar = min(qvar,opt.varThresh);
            
            %Scale the rhat and qhat terms by the variances and then add in
            %the chat and bhat terms
            rhat = rhat .* rvar;
            rhat = rhat + chat;
            qhat = qhat .* qvar;
            qhat = qhat + bhat;
            
            
            
        end
        
        
        
    end
    
    
    
    
    
    
    
end