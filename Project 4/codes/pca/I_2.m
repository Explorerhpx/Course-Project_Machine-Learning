load freyface.mat
X = double(X);  % 560（20*28）* 19165
%showfreyface(X(:,1:100))

N = size(X, 2);
[Vun, Dun] = eig(X*X'/N);
[lambda_un, order] = sort(diag(Dun));
Vun = Vun(:, order);

Xctr = X - repmat(mean(X, 2), 1, N);  %减去均值
[Vctr, Dctr] = eig(Xctr*Xctr'/N);
[lambda_ctr, order] = sort(diag(Dctr));
Vctr = Vctr(:, order);

%% 第1问
% subplot(2,2,1)
% plot(1:length(lambda_ctr),lambda_ctr)
% title('eigenspectra');
% subplot(2,2,2)
% plot(1:10,lambda_un((length(lambda_un)-9):(length(lambda_un)))); % 显示最后10个特征值
% title('Last ten eigenspectra');
% subplot(2,2,3)
% plot(0:19,lambda_un((length(lambda_un)-20):(length(lambda_un)-1))); % 显示最后10个特征值
% title('Last twenty to the last but one eigenspectra');

%% 第2问
% subplot(1,2,1)
% showfreyface(Vun(:,end-15:end));
% title('Uncentered')
% subplot(1,2,2)
% showfreyface(Vctr(:,end-15:end));
% title('Centered')

%% 第3问
%Y = Vctr(:,end-1:end)' * Xctr;  %中心化后投影到新空间（降到2维）
Y = Vun(:,end-1:end)' * X;  %投影到新空间（降到2维）
subplot(1,2,1)
plot(Y(1,:), Y(2,:), '.');
explorefreymanifold(Y, X);

%% 第4问
test = X(:,size(X,2)*ceil(rand(1)+0.01)); %随机选择一个点
origin = reshape(test,[20,28]); %原图
reconst1 =  Vctr(:,end-1:end) * Vctr(:,end-1:end)'* (test - mean(X, 2)) + mean(X, 2); %投影并重构
reconst1 = reshape(reconst1,[20,28]);
reconst2 =  Vctr(:,end-10:end) * Vctr(:,end-10:end)'* (test - mean(X, 2)) + mean(X, 2); %投影并重构
reconst2 = reshape(reconst1,[20,28]);
reconst3 =  Vctr(:,end-100:end) * Vctr(:,end-100:end)'* (test - mean(X, 2)) + mean(X, 2); %投影并重构
reconst3 = reshape(reconst1,[20,28]);
subplot(2,2,1)
imagesc(origin');
title('Origion')
subplot(2,2,2)
imagesc(reconst1');
title('Reconstruct dimension=2')
subplot(2,2,3)
imagesc(reconst2');
title('Reconstruct dimension=10')
subplot(2,2,4)
imagesc(reconst3');
title('Reconstruct dimension=100')
colormap gray;

%% 第5题
test = X(:,size(X,2)*ceil(rand(1)+0.01)); %随机选择一个点
origin = reshape(test,[20,28]); %原图
noise = imnoise(origin,'gaussian',0.01);    %加入高斯躁声 
recons =  Vctr(:,end-1:end) * Vctr(:,end-1:end)'* (noise(:) - mean(X, 2)) + mean(X, 2); %去噪并重构
recons = reshape(recons,[20,28]);
subplot(1,3,1)
imagesc(origin');
title('Origion')
subplot(1,3,2)
imagesc(noise');
title('Noisy picture')
subplot(1,3,3)
imagesc(recons');
title('Denoising picture')
colormap gray;
