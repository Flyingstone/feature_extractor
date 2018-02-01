function [m,label] = litekmeans(X, k)
% Perform k-means clustering.
%   X: d x n data matrix
%   k: number of seeds
% Written by Michael Chen (sth4nth@gmail.com).

rand('seed',1); %#ok<RAND>
n = size(X,2);
last = 0;
label = ceil(k*rand(1,n));  % random initialization
iter = 0;
tt = tic;
while any(label ~= last)
    iter = iter+1;
    fprintf('Iter %d:\n', iter);
    t_start = tic;
    [u,~,label] = unique(label);   % remove empty clusters
    k = length(u);
    E = sparse(1:n,label,1,n,k,n);  % transform label into indicator matrix
    m = X*(E*spdiags(1./sum(E,1)',0,k,k));    % compute m of each cluster
    last = label;
    [~,label] = max(bsxfun(@minus,m'*X,dot(m,m,1)'/2),[],1); % assign samples to the nearest centers
    fprintf('time: %f s diff %d\n', toc(t_start),sum(last~=label));
end
[~,~,label] = unique(label);
fprintf('total time: %f s\n', toc(tt));