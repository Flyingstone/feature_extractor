clear all;clc;
config;

addpath('.\yael_gmm');
addpath('.\yael_gmm\yael_gmm_mexw64');
addpath('.\utils');

path.idt = 'H:\Flickr_Hollywood2\Hollywood2\Hollywood2_features_L50_W16';
path.tgt = 'H:\Flickr_Hollywood2\extracted_features\Hollywood2_IDT_FV';

% compute GMM models
options.K                = GMM_codebook_size;
options.max_ite_kmeans   = 10;
options.max_ite_gmm      = 10;
options.gmm_flags_w      = 1;

options.seed             = 1234543;
options.num_threads      = 2;
options.verbose          = 0;


GMM_file_name = ['GMM_K',num2str(options.K),'.mat'];
if exist(GMM_file_name, 'file')
    fprintf('loading %s....\n', GMM_file_name);
    load(GMM_file_name, 'M', 'S', 'w', 'pca_projection', 'mean_feats');
else
    if ~exist('sample_data.mat', 'file')
        fprintf('prepare sample_data.mat!\n');
        return;
    end
    M = cell(desc_type_num,1);
    S = cell(desc_type_num,1);
    w = cell(desc_type_num,1);
    pca_projection = cell(desc_type_num,1);
    mean_feats = cell(desc_type_num,1);
    load('sample_data.mat', 'sample_features');
    rand('seed',1);
    rand_idx = randperm(size(sample_features,2));
    for fti = 1:desc_type_num        
        input_data = sample_features(desc_idx{fti},rand_idx(1:FV_sample_num));
        fprintf('PCA for the %d-th descriptor....\n',fti);
        [ProjectMatrix,~,mean_feat] = calc_pca(input_data);
        tgt_dim = length(desc_idx{fti})/2;
        pca_projection{fti} = ProjectMatrix(:,1:tgt_dim)';
        mean_feats{fti} = mean_feat;
        input_data = pca_projection{fti}*(input_data-repmat(mean_feat,1,size(input_data,2)));
        fprintf('computing GMM for the %d-th descriptor....\n',fti);
        [M{fti}, S{fti}, w{fti}]  = yael_gmm(input_data, options);
    end
    save(GMM_file_name, 'M', 'S', 'w', 'pca_projection', 'mean_feats', '-v7.3');
end

options.fv_flags_w = 1;
options.fv_flags_M = 1;
options.fv_flags_S = 1;
options.palpha     = 0.5; %(0,1) default value 0.5
options.norm_l2    = 1;

% loading data
[file_total_num, desc_total_num]= textread('stat.txt','%d\n%d');
file_total_num = file_total_num(1);
file_names = textread('feat_file_list.txt', '%s');
desc_nums = textread('feat_number_list.txt','%d');
desc_cumsum = cumsum(desc_nums);
assert(length(desc_nums)==file_total_num);

% generate fisher vectors
for ii = 1:file_total_num
    fprintf('processing %d/%d....\n',ii,file_total_num);
    
    file_name = file_names{ii};
    feat_file_name = fullfile(path.idt,file_name);

    save_file_name = fullfile(path.tgt,strrep(file_name,'.feat','.mat'));
    if ~exist(fileparts(save_file_name),'dir')
        mkdir(fileparts(save_file_name));
    end
    if exist(save_file_name, 'file')
        continue;
    end
    fv_feature = [];   %#ok<NASGU>
    flag = 0;
    while flag==0 && ~exist(save_file_name, 'file')
        try
            save(save_file_name, 'fv_feature', '-v7.3');
            flag = 1;
        catch %#ok<CTCH>
            continue;
        end
    end
    
    t_start = tic;
    
    % read file
    t1 = tic;
    decs = importdata(feat_file_name);
    decs = decs';
    
    fprintf('reading %f s\n', toc(t1));
    %f = options.K*(options.fv_flags_w + options.fv_flags_M*d + options.fv_flags_S*d); %dim of Fisher vector
    fv_feature = [];
    for fti = 1:desc_type_num
        t2 = tic;
        tmp_decs = decs(desc_idx{fti},:);
        tmp_decs = pca_projection{fti}*(tmp_decs-repmat(mean_feats{fti},1,size(tmp_decs,2)));
        p = yael_proba_gmm(tmp_decs, M{fti}, S{fti}, w{fti}, options);
        F = yael_fisher_gmm(tmp_decs, M{fti}, S{fti}, w{fti}, p, options);
        fv_feature = [fv_feature;F];
        fprintf('feature %d: %f s\n', fti, toc(t2));
    end
    
    flag = 0;
    while flag==0
        try
            save(save_file_name, 'fv_feature', '-v7.3');
            flag = 1;
        catch %#ok<CTCH>
            continue;
        end
    end
end
