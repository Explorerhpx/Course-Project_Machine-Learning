[Pi,mu,vary,logProbX] = mogEM_I(train2,2,40,1.0000e-08,200,0);
%[Pi,mu,vary,logProbX] = mogEM_I(train2,2,40,1.0000e-08,200,0);
subplot(2,2,1)
imagesc(reshape(mu(:,1),16,16));
title('mean')
subplot(2,2,2)
imagesc(reshape(vary(:,1),16,16));
title('variance');
subplot(2,2,3)
imagesc(reshape(mu(:,2),16,16));
title('mean')
subplot(2,2,4)
imagesc(reshape(vary(:,2),16,16));
title('variance');
colormap(gray);
Pi