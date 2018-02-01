%% xread_DT_bin
%  dt.trajectory hog hof mbhx mbhy
function [dt] = xread_DT_bin(file,nbin,grids)

nhog = nbin*grids(1)*grids(2)*grids(3);
nhof = (nbin+1)*grids(1)*grids(2)*grids(3);
nmbh = nbin*grids(1)*grids(2)*grids(3);
nfea =37 + nhog + nhof + 2*nmbh;

if exist(file, 'file')
    fid = fopen(file,'rb');
    temp = fread(fid, [nfea, inf],'single');
    fclose(fid);
    if ~isempty(temp)
    dt.trajectory = temp(8:37,:)';
    nstar = 38; nend = nstar + nhog -1;
    dt.hog = temp(nstar:nend,:)';
    nstar = nend + 1; nend = nend + nhof;
    dt.hof = temp(nstar:nend,:)';
    nstar = nend + 1; nend = nend + nmbh;
    dt.mbhx = temp(nstar:nend,:)';
    nstar = nend + 1; nend = nend + nmbh;
    dt.mbhy = temp(nstar:nend,:)';
    else 
        dt = [];
    end
else
    fprintf([file, 'invalidate, please check!']);
end