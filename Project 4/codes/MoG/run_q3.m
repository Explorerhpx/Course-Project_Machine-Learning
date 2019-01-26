load digits;
x = [train2, train3];
%-------------------- Add your code here --------------------------------
% Train a MoG model with 20 components on all 600 training vectors
% with both original initialization and your kmeans initialization. 

% 搜寻空间
t1 = clock;
% minvar= [4e-8,6e-8,8e-8,1e-7,3e-7,5e-7,7e-7];
% const = [4e-7,6e-7,8e-7,1e-6,3e-6,5e-6,7e-6];
minvar= [1e-7,1e-6,1e-5,1e-4,1e-3,0.01,0.1,1,10,100];
const = [1e-7,1e-6,1e-5,1e-4,1e-3,0.01,0.1,1,10,100];

train = [train2, train3];

result=zeros(length(minvar),length(const));
for x = 1:length(minvar)   % 在Minvarian 和 Randconst空间内搜索
    for y = 1:length(const)
        [Pi,mu,vary,logProbX] = mogEM_II(train3,10,20,minvar(x),const(y),0);
        result(x,y) = logProbX(end);
    end
end

% plot the result
[M,C]=meshgrid(log10(minvar),log10(const));
mesh(M,C,result);
x1 = xlabel('Minvar'); 
x2 = ylabel('Const');  
x3 = zlabel('LogProbability');
set(x1,'Rotation',27);  
set(x2,'Rotation',-30); 
hold on
set(gca,'xtick',log10([4e-8,7e-8,1e-7,5e-7,7e-7]),'xticklabel',{'4e-8','7e-8','1e-7','5e-7','7e-7'})
set(gca,'ytick',log10([4e-7,7e-7,1e-6,5e-6,7e-6]),'yticklabel',{'4e-7','7e-7','1e-6','5e-6','7e-6'})

% find the best parametr
[rows,cols] = find(result==max(max(result)));
minvar(rows)
const(cols)
t2 = clock;
etime(t2,t1)