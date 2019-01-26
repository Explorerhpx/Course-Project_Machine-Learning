% SWISS ROLL DATASET

N=2000;
K=12;
d=2;

clf; colordef none; colormap jet; set(gcf,'Position',[200,400,620,200]);

% PLOT TRUE MANIFOLD
tt0 = (3*pi/2)*(1+2*[0:0.02:1]); hh = [0:0.125:1]*30;
xx = (tt0.*cos(tt0))'*ones(size(hh));
yy = ones(size(tt0))'*hh;
zz = (tt0.*sin(tt0))'*ones(size(hh));
cc = tt0'*ones(size(hh));

subplot(3,2,1); cla;
surf(xx,yy,zz,cc);
view([12 20]); grid off; axis off; hold on;
lnx=-5*[3,3,3;3,-4,3]; lny=[0,0,0;32,0,0]; lnz=-5*[3,3,3;3,3,-3];
lnh=line(lnx,lny,lnz);
set(lnh,'Color',[1,1,1],'LineWidth',2,'LineStyle','-','Clipping','off');
axis([-15,20,0,32,-15,15]);

% GENERATE SAMPLED DATA
tt = (3*pi/2)*(1+2*rand(1,N));  height = 21*rand(1,N);
X = [tt.*cos(tt); height; tt.*sin(tt)];

% SCATTERPLOT OF SAMPLED DATA
subplot(3,2,2); cla;
scatter3(X(1,:),X(2,:),X(3,:),12,tt,'+');
view([12 20]); grid off; axis off; hold on;
lnh=line(lnx,lny,lnz);
set(lnh,'Color',[1,1,1],'LineWidth',2,'LineStyle','-','Clipping','off');
axis([-15,20,0,32,-15,15]); drawnow;

% RUN LLE ALGORITHM
Y=lle(X,K,d);

%% 1.3
% PCA 方法
Z = X;
N = size(Z,2);
Zctr = Z - repmat(mean(Z, 2), 1, N);  %减去均值
[Vctr, Dctr] = eig(Zctr*Zctr'/N);
[lambda_ctr, order] = sort(diag(Dctr));
Vctr = Vctr(:, order); %主成分分析

Z =  Vctr(:,end-1:end)'* Zctr; %投影到2维

% 使用工具包
tem = compute_mapping(X', 'NPE');
Z1 = tem';
tem = compute_mapping(X', 'SNE');
Z2 = tem';
%%


% SCATTERPLOT OF EMBEDDING
subplot(3,2,3); cla;
scatter(Y(1,:),Y(2,:),12,tt,'+');
title('LLE');
subplot(3,2,4); cla;
scatter(Z(1,:),Z(2,:),12,tt,'+');
title('PCA');
subplot(3,2,5); cla;
scatter(Z1(1,:),Z1(2,:),12,tt,'+');
title('NPE');
subplot(3,2,6); cla;
scatter(Z2(1,:),Z2(2,:),12,tt,'+');
title('SNE');
grid off;
set(gca,'XTick',[]); set(gca,'YTick',[]);