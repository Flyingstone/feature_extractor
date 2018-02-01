function extr_sift(img_dir, data_dir, image_list, image_index)
% for example
% img_dir = 'image/Caltech101';
% data_dir = 'data/Caltech101';

addpath('sift');

gridSpacing = 6;
patchSize = 16;
maxImSize = 300;
nrml_threshold = 1;

CalculateSiftDescriptor2(img_dir, data_dir, image_list, image_index, gridSpacing, patchSize, maxImSize, nrml_threshold);