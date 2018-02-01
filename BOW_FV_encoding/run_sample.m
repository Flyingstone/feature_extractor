clear all;clc;
config;

path.idt = 'H:\Flickr_Hollywood2\Hollywood2\Hollywood2_features_L50_W16';
sample_num = max(BOW_sample_num, FV_sample_num);

sample_dir = '.\sample_data';
[file_total_num, desc_total_num]= textread('stat.txt','%d\n%d');
file_total_num = file_total_num(1);
file_names = textread('feat_file_list.txt', '%s');

desc_nums = textread('feat_number_list.txt','%d');
desc_cumsum = [0; cumsum(desc_nums)];

assert(length(desc_nums)==file_total_num);

rand('seed',1); %#ok<RAND>
rand_idx = randperm(desc_total_num);
rand_idx = sort(rand_idx(1:sample_num),'ascend');
save(['rand_idx_',num2str(sample_num),'.mat'],'rand_idx','-v7.3');

if ~exist(sample_dir,'dir')
    mkdir(sample_dir);
end

sec_idx = zeros(desc_total_num,1);
FormatString = repmat('%f ',1, desc_len);

% sample descriptors
for fi = 1:file_total_num
    save_file_name = [sample_dir,'\descriptors_in_',num2str(fi),'.mat'];
    if exist(save_file_name, 'file')
        continue;
    end
    fprintf('processing %d/%d....\n',fi,file_total_num);

    feats = []; 
    flag = 0;
    while flag==0 && ~exist(save_file_name, 'file')
        try
            save(save_file_name, 'feats', '-v7.3');
            flag = 1;
        catch %#ok<CTCH>
            continue;
        end
    end
    
    select_desc_idx = rand_idx(rand_idx>=desc_cumsum(fi)+1 & rand_idx<=desc_cumsum(fi+1));
    search_idx = [1, select_desc_idx-desc_cumsum(fi)];
    
    % extract descriptor
    t_start = tic;
    fid = fopen(fullfile(path.idt,file_names{fi}),'r');
    for ii = 2:length(search_idx)
        feat = cell2mat(textscan(fid,FormatString,1,'HeaderLines',search_idx(ii)-search_idx(ii-1)))';
        feats = [feats feat];
    end
    fclose(fid);
    fprintf('time %f s\n',toc(t_start));
    

    flag = 0;
    while flag==0 
        try
            save(save_file_name, 'feats', '-v7.3');
            flag = 1;
        catch %#ok<CTCH>
            continue;
        end
    end
end

