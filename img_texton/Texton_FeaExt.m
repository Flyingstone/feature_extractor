function feaSet = Texton_FeaExt(I)
% extract Texton features from dense sampled positions
% I : input image
% feaSet: Texton features

% parameters
max_img_size = 300; % maximum image size

% pre-processing of image
if ndims(I) == 3
    I = im2double(rgb2gray(I));
else
    I = im2double(I);
end

[im_h, im_w] = size(I);

%different from 15Scenes
% if max(im_h, im_w) > max_img_size
%     I = imresize(I, max_img_size/max(im_h, im_w), 'bicubic');
%     [im_h, im_w] = size(I);
% end;  
  
responses = gaussfilter(I); % can be optimaize to speedup, only generate filter for one time
[m, n, o] = size(responses);

% dense sampling of positions
y_pos = 1:2:m;
x_pos = 1:2:n;

fea_num = length(y_pos)*length(x_pos); % number of features

feaSet.feaArr = zeros(o,fea_num);
feaSet.y = zeros(fea_num,1);
feaSet.x = zeros(fea_num,1);

index = 1;
for y = y_pos
    for x = x_pos
        feaSet.feaArr(:,index) = responses(y,x,:);
        feaSet.y(index) = y;
        feaSet.x(index) = x;
        index = index + 1;
    end
end

feaSet.width = n;
feaSet.height = m;
        
end
