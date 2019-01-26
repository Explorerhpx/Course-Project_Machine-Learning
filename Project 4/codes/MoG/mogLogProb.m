% function [logProb] = mogLogProb(p,mu,vary,x)
%
% 计算
% p:每个类的先验概率，mu:每个类的均值（一列一类），vary:每个类的方差（一列一类）
% x:数据点（一列一个点）
% Computes the log-probability of some data under a mixture of Gaussians
% with diagonal covariance matrices.
%
% Input:
%
%   Model parameters:
%
%    p(k) = prior probability of the kth cluster
%    mu(n,k) = the nth component of the mean for the kth cluster
%    vary(j,k) = variance of the jth dimension in the kth cluster
%
%   Data:
%
%    x(n,t) = the nth input for the tth training case
%  
% Output:
%
%    logProb(t) = the log-probability of the tth case using the current model
%

function [logProb] = mogLogProb(p,mu,vary,x)

K = length(p); N = length(x(:,1)); T = length(x(1,:));
% K: 类别数， N：样本维度， T:样本数目
ivary = 1./vary;
logProb = zeros(1,T);
for t=1:T;
  % Compute log P(c)p(x|c) and then log p(x)
  logPcAndx = log(p) - log(2*pi)*N/2 - 0.5*sum(log(vary'),2) ...
           - 0.5*sum(ivary.*(x(:,t)*ones(1,K)-mu).^2,1)';
  mx = max(logPcAndx); logProb(t) = log(sum(exp(logPcAndx-mx))) + mx;
  % 减mx应该是为了做一个normalization吧???
end;
