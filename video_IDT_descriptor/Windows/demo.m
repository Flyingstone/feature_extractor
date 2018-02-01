%DEMO
% extract improved dense trajectories
nbin = 8; grids = [2 2 3];
%	if you want to show the trajectories, please use -u 1
%	you can change the video paths so as to deal with all the videos 

video_path = 'sample.avi';
system(['DenseTrack.exe ' video_path ' -o ' video_path(1:end-4) '.bin -u 0']);
%	read all the idt features from .bin file by concatenating them
idt_con = xread_DT_bin_con([video_path(1:end-4) '.bin'],nbin,grids);
%	read idt features seperately
idt_sep = xread_DT_bin([video_path(1:end-4) '.bin'],nbin,grids);

pause;




