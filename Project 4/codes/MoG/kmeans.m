% Performs clustering using the k-means algorithm.
%
% Input:
%
%   x(:,t) = the N-dimensional training vector for the tth training case
%   K = number of clusters
%   iters = number of iterations of EM to apply
%
% Output:
%
%   means(:, c) = mean of the cth cluster


function [means] = kmeans(x, K, iters)
    % Initialize the means to K random training vectors
    % 随机挑选K个点作为K个中心
    perm = randperm(size(x, 2));
    means = x(:, perm(1:K));
    
    N = size(x, 2);  % 样本数目
    for iter=1:iters
        dist = nan(K, N);
        for k=1:K
            dist(k, :) = distmat(x', means(:, k)');
        end
        [m, class] = min(dist); % 给每个点分类
        for k=1:K
            means(:,k) = mean(x(:, class == k), 2);  % 中心更新
        end
    end
