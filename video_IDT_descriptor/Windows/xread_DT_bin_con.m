%% xread_DT_bin
%  dt.trajectory hog hof mbhx mbhy
function [dt] = xread_DT_bin_con(file,nbin,grids)

nhog = nbin*grids(1)*grids(2)*grids(3);
nhof = (nbin+1)*grids(1)*grids(2)*grids(3);
nmbh = nbin*grids(1)*grids(2)*grids(3);
nfea =37 + nhog + nhof + 2*nmbh;

if exist(file, 'file')
    fid = fopen(file,'rb');
    dt = fread(fid, [nfea, inf],'single');
    fclose(fid);
else
    fprintf([file, 'invalidate, please check!']);
end