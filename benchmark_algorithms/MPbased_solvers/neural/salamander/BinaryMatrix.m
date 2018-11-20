% BinaryPackMatrix:  Packed matrix for binary data
classdef BinaryMatrix < handle
    
    properties
        nrow;   % number of rows
        ncol;   % number of cols
        
        ncolPack;       % number of columns in packed form
        wordSize = 32;  % number of bits per word
        
        % Packed matrix:  A(irow,icol) = bitget( A(irow,icol1), ibit )
        % where icol1 = floor(icol/wordSize)+1 and 
        % ibit = mod(icol-1,wordSize)+1
        Apack;      
        
        % To simplify computations, some inputs can be ignored
        % if they fall below a certain tolerance level
        relTol = 0.0;   % Do not filter if input < relTol*max(input)
        absTol = 0.0;   % Don not filter if norm(input) < absTol
        
        % Row map:  with repeated rows, row A(irow,:) is stored
        % in Apack(rowMap(irow),:).
        rowMap = [];
    end
    
    methods
        % Constructor:  This constructor has two forms
        %
        % BinaryMatrix(nrow, ncol): creates an empty binary packed
        % matrix of dimension nrow x ncol
        %
        % BinaryMatrix(rowMap, ncol):  creates an binary packed
        % matrix of dimension of max(rowMap) x ncol
        function obj = BinaryMatrix(arg1, ncol)
            if (length(arg1) == 1)                
                obj.nrow = arg1;
                nrowPack = obj.nrow;
            else
                obj.nrow = length(arg1);
                obj.rowMap = arg1;
                nrowPack = max(obj.rowMap);
            end                
            obj.ncol = ncol;
            obj.ncolPack = ceil(obj.ncol/obj.wordSize);
            obj.Apack = zeros(nrowPack, obj.ncolPack,'uint32');
        end
        
        % addRow:  Adds a row to the matrix
        function setRow(obj, irow, arow)
            % Pack the row
            arowPack = zeros(1,obj.ncolPack, 'uint32');
            for icol = 1:obj.ncol
                icolp = floor((icol-1)/obj.wordSize) + 1;
                ibit = mod((icol-1), obj.wordSize) + 1;
                arowPack(icolp) = bitset( arowPack(icolp),ibit, arow(icol));                                
            end
            
            % Set the row
            obj.Apack(irow, :) = arowPack;
        end
        
        % getCol:  Gets a column.  Irow is a subset of rows.
        % If Irow = [], then the function returns all rows.
        % If Irow < 0, then the functions returns the rows in packed
        % form.  
        function acol = getCol(obj, icol, Irow)
            if (nargin < 3)
                Irow = (1:obj.nrow)';
            end
            icolp = floor((icol-1)/obj.wordSize) + 1;
            ibit = mod((icol-1), obj.wordSize) + 1;
            acolPack = double( bitget(obj.Apack(:,icolp), ibit) );
            if (any(Irow < 0))
                acol = acolPack;
            elseif isempty(obj.rowMap) 
                acol = acolPack(Irow);
            else
                acol = acolPack(obj.rowMap(Irow));
            end
        end
        
        % setCol:  Sets a column
        function setCol(obj, icol, colval)
            icolp = floor((icol-1)/obj.wordSize) + 1;
            ibit = mod((icol-1), obj.wordSize) + 1;
            obj.Apack(:,icolp) = bitset(obj.Apack(:,icolp), ibit, colval);
        end
        
        % firFilt:  FIR filter of the columns with H
        % The operation is equivalent to implementing the matrix
        % multiply  y = T*h where h = H(:) and T is the matrix
        %
        %    T = [T_1 ... T_ncol]
        %
        % with T_j being the Toeplitz matrix with A(:,j) as its first 
        % column.
        function y = firFilt(obj, H, Irow)
            if (nargin < 3)
                Irow = (1:obj.nrow)';
            end
            
            % Find max column norm
            Hnorm = sqrt( sum(abs(H).^2) ); 
            Hmax = max(Hnorm);
            
            % Loop over columns
            ny = length(Irow);
            y = zeros(ny,1);
            for icol = 1:obj.ncol
                if ((Hnorm(icol) <= obj.absTol) || (Hnorm(icol) <= obj.relTol*Hmax))
                    continue;
                end
                acol = obj.getCol(icol,Irow);
                y1 = conv(acol, H(:,icol));
                y = y + y1(1:ny);
            end            
        end
        
        % firFiltTr:  FIR filter with the rows of A
        % The operation is equivalent to h = T'*y where T
        % is the matrix in firFilt and 
        %   H = reshape(h, ndly, ncol)
        function H = firFiltTr(obj,y,ndly,Irow,dispNum)
            if (nargin < 4)
                Irow = (1:obj.nrow)';
            end
            if (isempty(Irow))
                Irow = (1:obj.nrow)';
            end            
            if (nargin < 5)
                dispNum = 0;
            end
            ny = length(Irow);
            H = zeros(ndly,obj.ncol);
            for icol = 1:obj.ncol
                if ((dispNum > 0) && (mod(icol,dispNum)==0))
                    fprintf(1, '%d of %d\n' ,icol, obj.ncol);
                end
                acol = obj.getCol(icol,Irow);
                for idly = 1:ndly
                    H(idly,icol) = acol(1:ny-idly+1)'*y(idly:ny);
                end
                %h = conv(acol,flipud(y));
                %H(:,icol) = h(obj.nrow:-1:obj.nrow-ndly+1);
            end            
        end
        
        % Set tolerance levels
        function setTol(obj, relTol, absTol)
            obj.relTol = relTol;
            obj.absTol = absTol;
        end
        
        % Create submatrix with columns
        function objRed = getSubMatrix(obj,Icol)
            
            % Create a matrix with the reduced dimensions
            ncolRed = length(Icol);
            if (isempty(obj.rowMap))
                objRed = BinaryMatrix(obj.nrow, ncolRed);
            else
                objRed = BinaryMatrix(obj.rowMap, ncolRed);
            end

            % Fill the columns
            for icol = 1:ncolRed
                acol = obj.getCol(Icol(icol), -1); % Get row in packed form
                objRed.setCol(icol, acol);
            end
        end
    end
        
    
end
