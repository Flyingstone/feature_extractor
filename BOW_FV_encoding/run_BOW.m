clear all;clc;
config;

addpath('.\utils');
path.idt = 'H:\Flickr_Hollywood2\Hollywood2\Hollywood2_features_L50_W16';
path.tgt = 'H:\Flickr_Hollywood2\extracted_features\Hollywood2_IDT_BOW';

file_names = textread('feat_file_list.txt', '%s');

% prepare codebook
if exist('BOW_codebook.mat', 'file')
    fprintf('loading BOW_codebook....\n');
    load('BOW_codebook.mat', 'BOW_codebook');
else    
    if ~exist('sample_data.mat', 'file')
        fprintf('prepare sample_data.mat!\n');
        return;
    end
    load('sample_data.mat', 'sample_features');
    rand('seed',1);
    rand_idx = randperm(size(sample_features,2));
    BOW_codebook = cell(desc_type_num,1);
    for tpi = 1:desc_type_num
        smpl = sample_features(desc_idx{tpi},rand_idx(1:BOW_sample_num));
        BOW_codebook{tpi} = litekmeans(smpl, BOW_codebook_size);
    end
    save('BOW_codebook.mat', 'BOW_codebook', '-v7.3');
end

% quantize descriptors
for ii = 1:length(file_names)
    fprintf('processing %d/%d....\n',ii,length(file_names));
    
    file_name = file_names{ii};
    
    feat_file_name = fullfile(path.idt,file_name);    
    save_file_name = fullfile(path.tgt,strrep(file_name,'.feat','.mat'));
    if ~exist(fileparts(save_file_name),'dir')
        mkdir(fileparts(save_file_name));
    end
    if exist(save_file_name, 'file')
        continue;
    end
    feat = [];  %#ok<NASGU>
    flag = 0;
    while flag==0 && ~exist(save_file_name, 'file')
        try
            save(save_file_name, 'feat', '-v7.3');
            flag = 1;
        catch %#ok<CTCH>
            continue;
        end
    end
    
    t_start = tic;
    
    % quantize descriptors
    descs = importdata(feat_file_name);
    descs = descs';    

    feat = cell(desc_type_num,1);
    for fti = 1:desc_type_num
        part_desc = descs(desc_idx{fti},:);
        [~,clst_lbl] = max(bsxfun(@minus,BOW_codebook{fti}'*part_desc,dot(BOW_codebook{fti},BOW_codebook{fti},1)'/2),[],1); 
        feat{fti} = hist(clst_lbl, 1:BOW_codebook_size)';        
    end 
    feat = cell2mat(feat);  
    fprintf('time %f s\n',toc(t_start));

    flag = 0;
    while flag==0 
        try
            save(save_file_name, 'feat', '-v7.3');
            flag = 1;
        catch %#ok<CTCH>
            continue;
        end
    end
end


