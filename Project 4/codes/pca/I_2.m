load freyface.mat
X = double(X);  % 560��20*28��* 19165
%showfreyface(X(:,1:100))

N = size(X, 2);
[Vun, Dun] = eig(X*X'/N);
[lambda_un, order] = sort(diag(Dun));
Vun = Vun(:, order);

Xctr = X - repmat(mean(X, 2), 1, N);  %��ȥ��ֵ
[Vctr, Dctr] = eig(Xctr*Xctr'/N);
[lambda_ctr, order] = sort(diag(Dctr));
Vctr = Vctr(:, order);

%% ��1��
% subplot(2,2,1)
% plot(1:length(lambda_ctr),lambda_ctr)
% title('eigenspectra');
% subplot(2,2,2)
% plot(1:10,lambda_un((length(lambda_un)-9):(length(lambda_un)))); % ��ʾ���10������ֵ
% title('Last ten eigenspectra');
% subplot(2,2,3)
% plot(0:19,lambda_un((length(lambda_un)-20):(length(lambda_un)-1))); % ��ʾ���10������ֵ
% title('Last twenty to the last but one eigenspectra');

%% ��2��
% subplot(1,2,1)
% showfreyface(Vun(:,end-15:end));
% title('Uncentered')
% subplot(1,2,2)
% showfreyface(Vctr(:,end-15:end));
% title('Centered')

%% ��3��
%Y = Vctr(:,end-1:end)' * Xctr;  %���Ļ���ͶӰ���¿ռ䣨����2ά��
Y = Vun(:,end-1:end)' * X;  %ͶӰ���¿ռ䣨����2ά��
subplot(1,2,1)
plot(Y(1,:), Y(2,:), '.');
explorefreymanifold(Y, X);

%% ��4��
test = X(:,size(X,2)*ceil(rand(1)+0.01)); %���ѡ��һ����
origin = reshape(test,[20,28]); %ԭͼ
reconst1 =  Vctr(:,end-1:end) * Vctr(:,end-1:end)'* (test - mean(X, 2)) + mean(X, 2); %ͶӰ���ع�
reconst1 = reshape(reconst1,[20,28]);
reconst2 =  Vctr(:,end-10:end) * Vctr(:,end-10:end)'* (test - mean(X, 2)) + mean(X, 2); %ͶӰ���ع�
reconst2 = reshape(reconst1,[20,28]);
reconst3 =  Vctr(:,end-100:end) * Vctr(:,end-100:end)'* (test - mean(X, 2)) + mean(X, 2); %ͶӰ���ع�
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

%% ��5��
test = X(:,size(X,2)*ceil(rand(1)+0.01)); %���ѡ��һ����
origin = reshape(test,[20,28]); %ԭͼ
noise = imnoise(origin,'gaussian',0.01);    %�����˹���� 
recons =  Vctr(:,end-1:end) * Vctr(:,end-1:end)'* (noise(:) - mean(X, 2)) + mean(X, 2); %ȥ�벢�ع�
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
