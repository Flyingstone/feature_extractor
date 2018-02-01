function ExtractVideoFeatures_Youtube(nPart, m, n)

nPart = 1;
m = 1;
n = 1;

datapath = 'F:\Youtube\Events';
savepath = 'D:\Lixin\ExtractVideoFeatures\FEATURES\Youtube';

events = dir(fullfile(datapath));
events = events(3:end);

avsfilename = [num2str(nPart), '_Youtube.avs'];
for i = m:n
    
    eventpath     = fullfile(datapath, events(i).name);
    saveeventpath = fullfile(savepath, events(i).name);
    
    mkdir(saveeventpath);
    
    tmp = dir(eventpath);
    tmp = tmp(3:end);
    tmp = struct2cell(tmp);
    
    vdofiles = tmp(1, :);
    
    if ~isempty(intersect(vdofiles, 'Thumbs.db'))
        fprintf(2, 'Thumbs.db in %s.\n', events(i).name);
    end
    
    for j = 1:length(vdofiles)
        inputstring1 = ['movie = DirectShowSource("', fullfile(eventpath, vdofiles{j}), '")'];
        inputstring2 = 'return LanczosResize(ConvertToRGB24(movie), 160, 120)';
        fid = fopen(avsfilename, 'w');
        fprintf(fid, '%s\r\n', inputstring1);
        fprintf(fid, '%s\r\n', inputstring2);
        fclose(fid);

        [pathstr, name, ext, versn] = fileparts(vdofiles{j});
        
        dos(['D:\Lixin\ExtractVideoFeatures\stip-1.0-winlinux\bin\stipdet.exe -f ', avsfilename, ' -o ', fullfile(savepath, [name, '.hog'])]);
    end
end