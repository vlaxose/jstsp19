% function runRealMatComp_perf
%%************************************************************************
%% run matrix completion problems for real data sets.
%%
%% NNLS, version 0:
%% Copyright (c) 2009 by
%% Kim-Chuan Toh and Sangwoon Yun
%%
%% modified by Zaiwen Wen
%%************************************************************************

clear;
warning off


%addpath('/home/wenzw/work/MatrixCmp/code/NNLS-0/MatCompData')


for dataset = [1:7]    
    if (dataset == 1)
        exfile = 'jester-data-1';
        Dt     = xlsread(exfile);
        fname = 'jester-1';
    elseif (dataset == 2)
        exfile = 'jester-data-2';
        Dt     = xlsread(exfile);
        fname = 'jester-2';
    elseif (dataset == 3)
        exfile = 'jester-data-3';
        Dt     = xlsread(exfile);
        fname = 'jester-3';
    elseif (dataset == 4)
        %%
        %% jester dataset
        %%
        exfile1 = 'jester-data-1';
        Dt1     = xlsread(exfile1);
        exfile2 = 'jester-data-2';
        Dt2     = xlsread(exfile2);
        exfile3 = 'jester-data-3';
        Dt3     = xlsread(exfile3);
        Dt = [Dt1;Dt2;Dt3];
        fname = 'jester-all';
    elseif (dataset == 5)
        %%
        %% MovieLens Dataset: 943x1682 (usersxmovies)
        %%
        tM = dlmread('u100K.data','');
        M  = spconvert(tM);
        fname = 'moive-100K';
    elseif (dataset == 6)
        %%
        %% MovieLens Dataset: 6040x3900 (usersxmovies)
        %%
        tM = dlmread('u1M.data','');
        M  = spconvert(tM);
        fname = 'moive-1M';
    elseif (dataset == 7)
        %%
        %% MovieLens Dataset: 71567x10681 (usersxmovies)
        %%
        tM = dlmread('u10M.data','');
        M  = spconvert(tM);
        fname = 'moive-10M';
    end
    if (dataset <= 4)
        %%
        %% jester data set
        %%
        M = Dt(1:end,2:end);
        [nr,nc] = size(M);
        %%
        %% make zero ratings slightly non-zero.
        %%
        [ii,jj] = find(M==0);
        M = M + spconvert([ii,jj,1e-8*ones(length(ii),1);nr,nc,0]);
        %%
        %% set non-rated entries (currently equal to 99) to 0.
        %%
        [ii,jj] = find(M==99);
        M = M - spconvert([ii,jj,99*ones(length(ii),1); nr,nc,0]);

        %%
        NumPerRow = 10;
        Msub = zeros(nr,nc);
        for i=1:nr
            rand('state',i);
            Mrow = M(i,:);
            truerow = find( abs(Mrow) > 1e-13 );
            len     = length(truerow);
            rp      = randperm(len);
            chrow   = truerow(rp(1:min(NumPerRow,len)));
            Msub(i,chrow) = Mrow(chrow);
        end
        colnorm = sum(Msub.*Msub);
        colnormidx = find(colnorm == 0);
        if ~isempty(colnormidx)
            for j = 1:length(colnormidx)
                rand('state',j);
                jcol = colnormidx(j);
                Mcol = M(:,jcol);
                truecol = find( abs(Mcol) > 1e-13 );
                len     = length(truecol);
                rp      = randperm(len);
                chcol   = truecol(rp(1:min(NumPerRow,len)));
                Msub(chcol,jcol) = Mcol(chcol);
            end
        end
    else
        %%
        %% MovieLens data set
        %%
        normM = sum(M.*M);
        idx = find(normM == 0);
        M = M(:,setdiff([1:size(M,2)],idx));  %% removed zero columns.
        %fprintf('\n***** removed %2.1d zero columns in M',length(idx));
        Mt = M';
        [nr,nc] = size(M);
        nnzM = nnz(M);
        NumElement = 10;
        if (NumElement > 0)
            rr = zeros(nnzM,1);
            cc = zeros(nnzM,1);
            vv = zeros(nnzM,1);
            count = 0;
            for i=1:nr
                rand('state',i);
                Mrow = full(Mt(:,i))';
                truerow = find( abs(Mrow) > 1e-13);
                len     = length(truerow);
                rp      = randperm(len);
                NumPerRow = min(max(NumElement,floor(len*0.5)),len);
                %% NumPerRow = min(NumElement,len);
                chrow = truerow(rp(1:NumPerRow));
                idx = [1:NumPerRow];
                rr(count+idx) = i*ones(NumPerRow,1);
                cc(count+idx) = chrow;
                vv(count+idx) = Mrow(chrow);
                count = count + NumPerRow;
                %if (rem(i,1000)==0); fprintf('\n i = %2.1d',i); end
            end
            rr = rr(1:count); cc = cc(1:count); vv = vv(1:count);
            Msub = spconvert([rr,cc,vv; nr,nc,0]);
            colnorm = sum(Msub.*Msub);
            colnormidx = find(colnorm == 0);
            if ~isempty(colnormidx)
                rr = zeros(nnzM,1);
                cc = zeros(nnzM,1);
                vv = zeros(nnzM,1);
                count = 0;
                for j = 1:length(colnormidx)
                    rand('state',j);
                    jcol = colnormidx(j);
                    Mcol = full(M(:,jcol));
                    truecol = find( abs(Mcol) > 1e-13 );
                    len     = length(truecol);
                    rp      = randperm(len);
                    NumPerCol = min(max(NumElement,floor(len*0.5)),len);
                    %% NumPerCol = min(NumElement,len);
                    chcol = truecol(rp(1:NumPerCol));
                    idx = [1:NumPerCol];
                    rr(count+idx) = chcol;
                    cc(count+idx) = jcol*ones(NumPerCol,1);
                    vv(count+idx) = Mcol(chcol);
                    count = count + NumPerCol;
                end
                rr = rr(1:count); cc = cc(1:count); vv = vv(1:count);
                Msub = Msub + spconvert([rr,cc,vv; nr,nc,0]);
            end
        else
            Msub = M;
        end
    end

    save(fname,'fname','M','nr','nc','Msub');




end 
