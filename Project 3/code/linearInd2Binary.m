function [y] = linearInd2Binary(ind,nLabels)
% ��yת��Ϊ-1,1��ʾ������
% eg�� 2 = [-1,1,-1,-1,-1,-1,-1,-1,-1]
% 4 = [-1,-1,-1,1 ����]
% [3,5] = [-1,-1,1,-1,-1,-1����
%          -1,-1,-1,-1,1,-1����]
n = length(ind);

y = -ones(n,nLabels);

for i = 1:n
    y(i,ind(i)) = 1;
end