%% Example 1: Fit a elliptical MVGM 

clear,close all
addpath('.\yael_gmm_mexw32');
addpath('.\yael_gmm_mexw64');
d                                     = 2;
N                                     = 5000;
K_true                                = 8;

P_true                                = permute((1/K_true)*ones(1 , K_true) , [1 3 2]);
M_true                                = permute([1.5 , 1 , 0 , -1 , -1.5 , -1 , 0 , 1 ; 0 1 , 1.5 , 1 , 0 , -1 , -1.5 , -1] , [1 3 2]);
S_true                                = cat(3 , diag([0.01 , 0.1]) , diag([0.1 , 0.1]) , diag([0.1 , 0.01]) , diag([0.1 , 0.1]) , diag([0.01 , 0.1]) , diag([0.1 , 0.1]) , diag([0.1 , 0.01]) , diag([0.1 , 0.1]));
Z                                     = single(sample_mvgm(N , M_true , S_true , P_true));


options.K                            = 8;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 10;
options.gmm_flags_w                  = 1;

options.seed                         = 1234543;
options.num_threads                  = 2;
options.verbose                      = 0;


tic,[M , S , w]                      = yael_gmm(Z  , options);,toc
p                                    = yael_proba_gmm(Z , M , S , w , options);

options.fv_flags_w                   = 1;
options.fv_flags_M                   = 1;
options.fv_flags_S                   = 1;
options.palpha                       = 0.5; %(0,1) default value 0.5
options.norm_l2                      = 1;

%dim Fisher vector
f                                    = options.K*(options.fv_flags_w + options.fv_flags_M*d + options.fv_flags_S*d);

F                                    = yael_fisher_gmm(Z , M , S , w , p , options);


M                                    = reshape(M , [2 , 1 , options.K]);
S                                    = reshape([S(1 , :) ; zeros(2 , options.K) ; S(2 , :)] , [2 , 2 , options.K]);

[x , y]                              = ndellipse(double(M) , double(S));

figure(1)
plot(Z(1 , :) , Z(2 , :) , 'k+' , 'markersize' , 2 , 'linewidth' , 2);
hold on
plot(M(1 , :) , M(2 , :) , 'mo' , x , y , 'r', 'linewidth' , 2 , 'markersize' , 6);
% h = voronoi(double(M(1 , :)) , double(M(2 , :)) );
% set(h ,  'linewidth' , 2);
h = title(sprintf('GMM fitting for elliptical data, K = %d' , options.K));
set(h ,  'fontsize' , 12);
hold off
grid on
%axis square
drawnow

figure(2)
plot(F)
h = title(sprintf('Fisher vector for elliptical data, K = %d, d = %d, dim = %d' , options.K , d , f));
set(h ,  'fontsize' , 12);
drawnow


%% Example 2: Fit spiral data 

clear
d                                    = 2;
N                                    = 5000;
Z                                    = single(spiral2d(N));


options.K                            = 60;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 10;
options.init_random_mode             = 0;
options.redo                         = 1;
options.normalize_sophisticated_mode = 0;
options.BLOCK_N1                     = 1024;
options.BLOCK_N2                     = 1024;
options.gmm_1sigma                   = 0;
options.gmm_flags_no_norm            = 0;
options.gmm_flags_w                  = 1;

options.seed                         = 1234543;
options.num_threads                  = 2;
options.verbose                      = 0;


tic,[M , S , w]                      = yael_gmm(Z  , options);,toc

M                                    = reshape(M , [2 , 1 , options.K]);
S                                    = reshape([S(1 , :) ; zeros(2 , options.K) ; S(2 , :)] , [2 , 2 , options.K]);

[x , y]                              = ndellipse(double(M) , double(S));

figure(3)
plot(Z(1 , :) , Z(2 , :) , 'k+' , 'markersize' , 2 , 'linewidth' , 2);
hold on
plot(M(1 , :) , M(2 , :) , 'mo' , x , y , 'r', 'linewidth' , 2 , 'markersize' , 6);
h = title(sprintf('GMM fitting for spiral data, K = %d' , options.K));
set(h ,  'fontsize' , 12);
hold off
grid on
axis square
drawnow


%% Example 3: GMM for SIFT patches + Fisher Vectors

clear


options.scale                        = [1];
options.sigma_scale                  = 0.6;
options.deltax                       = 100;
options.deltay                       = 100;
options.color                        = 0;
options.nori                         = 8;
options.alpha                        = 9;
options.nbins                        = 4;
options.patchsize                    = 9;
options.norm                         = 4;
options.clamp                        = 0.2;
options.rmmean                       = 0;


options.sigma_edge                   = 1;
[options.kernely , options.kernelx]  = gen_dgauss(options.sigma_edge);
options.weightx                      = gen_weight(options.patchsize , options.nbins);
options.weighty                      = options.weightx';


I                                    = imread('image_0001.jpg');
X                                    = denseSIFT(I , options); 

I                                    = imread('image_0174.jpg');
X                                    = single([X , denseSIFT(I , options)]); 



[d , N]                              = size(X);



options.K                            = 64;
options.max_ite_kmeans               = 10;
options.max_ite_gmm                  = 10;
options.gmm_flags_w                  = 1;

options.seed                         = 1234543;
options.num_threads                  = -1;
options.verbose                      = 0;


tic,[M , S , w]                      = yael_gmm(X  , options);,toc
p                                    = yael_proba_gmm(X , M , S , w , options);

options.fv_flags_w                   = 1;
options.fv_flags_M                   = 1;
options.fv_flags_S                   = 1;
options.palpha                       = 0.0;
options.norm_l2                      = 0;

%dim Fisher vector
f                                    = options.K*(options.fv_flags_w + options.fv_flags_M*d + options.fv_flags_S*d);

mF                                   = yael_fisher_gmm(X , M , S , w , p , options);
sF                                   = yael_fisher_gmm(X , M , S , w , p , options , mF);


%test image 
options.palpha                       = 0.5;
options.norm_l2                      = 1;

I                                    = imread('image_0010.jpg');
X                                    = single(denseSIFT(I , options)); 
p                                    = yael_proba_gmm(X , M , S , w , options);
F                                    = yael_fisher_gmm(X , M , S , w , p , options , mF , sF);

Fori                                 = yael_fisher_gmm_ori(X , M , S , w , p , options);


figure(4)
imagesc(M)
h = title(sprintf('Mean vectors trained by GMM, K = %d, d = %d, dim = %d' , options.K , d , f));
set(h ,  'fontsize' , 12);
drawnow


figure(5)
plot(F)
h = title(sprintf('Fisher vector for SIFT patches, K = %d, d = %d, \\alpha = %2.2f, dim = %d' , options.K , d , options.palpha, f));
set(h ,  'fontsize' , 12);
axis([0 , f+1 , 1.1*min(F) , 1.1*max(F)])
drawnow

figure(6)
plot(Fori)
h = title(sprintf('Fisher vector (from yeal package) for SIFT patches, K = %d, d = %d, \\alpha = %2.2f, dim = %d' , options.K , d , options.palpha, f));
set(h ,  'fontsize' , 12);
axis([0 , f+1 , 1.1*min(Fori) , 1.1*max(Fori)])
drawnow


figure(7)
hist(F , 100)
h = title(sprintf('distribution of Fisher vector values, K = %d, d = %d, dim = %d' , options.K , d , f));
set(h ,  'fontsize' , 12);



