function readMovieLenData

% data is from the website:
% http://www.grouplens.org/system/files/ml-data-10M100K.tar.gz

fid = fopen('/home/wenzw/work/MatrixCmp/code/NNLS-0/MatCompData/ratings.dat', 'r');
fidw = fopen('/home/wenzw/work/MatrixCmp/code/NNLS-0/MatCompData/u10M.data', 'w');

for di = 1: 10000054
% for di = 1: 1000
    a = fscanf(fid, '%d', 1);    fscanf(fid, '%c', 2);
    b = fscanf(fid, '%d', 1);    fscanf(fid, '%c', 2);
    c = fscanf(fid, '%f', 1);    fscanf(fid, '%c', 2);  fscanf(fid, '%d', 1);
    fprintf(fidw, '%u \t %u \t %g \n', a, b, c);
end

fclose(fid);
fclose(fidw);


% fid = fopen('/home/wenzw/work/MatrixCmp/code/NNLS-0/MatCompData/u.data', 'r');
% fidw = fopen('/home/wenzw/work/MatrixCmp/code/NNLS-0/MatCompData/u100K2.data2', 'w');
% 
% for di = 1: 100000
% % for di = 1: 1000
%     a = fscanf(fid, '%d', 1);    
%     b = fscanf(fid, '%d', 1);    
%     c = fscanf(fid, '%f', 1);    fscanf(fid, '%d', 1);
%     fprintf(fidw, '%u \t %u \t %g \n', a, b, c);
% end
% 
% fclose(fid);
% fclose(fidw);