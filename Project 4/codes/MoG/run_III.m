clear all
load digits

% 搜寻空间
minvar= [1e-7,1e-6,1e-5,1e-4,1e-3,0.01,0.1,1,10,100];
const = [1e-7,1e-6,1e-5,1e-4,1e-3,0.01,0.1,1,10,100];

result=zeros(length(minvar),length(const));
for x = 1:length(minvar)   % 在Minvarian 和 Randconst空间内搜索
    for y = 1:length(const)
        [Pi,mu,vary,logProbX] = mogEM_I(train3,2,40,minvar(x),const(y),0);
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
plot3(M,C,result,'x','MarkerSize',3);
set(gca,'xtick',log10([1e-7,1e-5,1e-3,0.1,10,100]),'xticklabel',{'1e-7','1e-5','1e-3','0.1','10','100'})
set(gca,'ytick',log10([1e-7,1e-5,1e-3,0.1,10,100]),'yticklabel',{'1e-7','1e-5','1e-3','0.1','10','100'})

% find the best parametr
[rows,cols] = find(result==max(max(result)));
minvar(rows)
const(cols)