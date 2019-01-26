#2.2 2.3
#2.3_1

Shuju<-read.table("C:\\Users\\DELL\\Desktop\\课件与课本\\3上\\统计（机器）学习\\pj\\project1\\data\\prostate.data.txt",header=1)

Shuju<-data.matrix(Shuju)

#分开参数和结果
y<-Shuju[,9]
X<-Shuju[,-9]

#分开训练集与测试集
chouqu<-sample(1:97,50)
X_tr<-X[chouqu,];
y_tr<-y[chouqu];
chouqu<-chouqu*(-1);
X_te<-X[chouqu,];
y_te<-y[chouqu]

#保存原始数据
X_tr_o<-X_tr
y_tr_o<-y_tr
X_te_o<-X_te
y_te_o<-y_te


#standerdizaion
S<-vector()    #标准差集
M<-vector()    #均值集
for(a in 1:8){   #对元素进行standerdise
  S[a]<-sd(X_tr[,a]);
  M[a]<-mean(X_tr[,a]);
  X_tr[,a]<-(X_tr[,a]-M[a])/S[a];
}

My<-mean(y_tr);
y_tr<-(y_tr-My);


#岭回归
ridge<-function (X, y, d2){
  X_t<-t(X)%*%X;
  Nl<-dim(X_t)[2];
  X_t<-X_t+diag(Nl)*d2;
  W<-solve(X_t)%*%t(X)%*%y;
  W
}

#作图
D2<-seq(-1.5,3.5,0.1)
D2<-10^D2
Der<-matrix(nrow=8,ncol=length(D2));
for(i in 1:length(D2)){
  Der[,i]<-ridge(X_tr,y_tr,D2[i])
}
#对坐标轴进行伸缩
D2_p<-log10(D2)
with(iris, plot(main="Regularization path for ridge regression",xlim=c(-2,4),ylim=c(-0.4,0.8),D2_p,Der[1,],type="l",col=1,pch=20,cex=1,xlab="δ2",ylab="θ",xaxt="n",yaxs="i",xaxs="i"))
grid(nx=6,ny=6,lwd=1,lty=2,col="blue") 

for(i in 2:8) lines(D2_p,Der[i,],col=i,type='l',pch=20)
axis(side=1,at = c(-2,-1,0,1,2,3,4),labels = c('10^-2','10^-1','10^0','10','10^2','10^3','10^4'))   

#图例
legend(x=3.0,y=0.8,c('lcavol','lweight','age','lbph','svi','lcp','gleason','pgg45'),col=1:8,lty=1)


#=================================================================
#2.3_2
#计算出W0后计算training error 与 test error
ridge_2<-function (X, y, d2){
  X_t<-t(X)%*%X;
  Nl<-dim(X_t)[2];
  X_t<-X_t+diag(Nl)*d2;
  W<-solve(X_t)%*%t(X)%*%y;

  #计算W0
  W<-W[,1];
  W_fuzhu<-vector();
  for(i in 1:8){
    W_fuzhu[i]=mean(X[,i]);
  }
  W0<-mean(y)-t(W)%*%W_fuzhu;
  W<-c(W0,W);
  W
}

#standerdizaion
for(a in 1:8){
  X_te[,a]<-(X_te[,a]-M[a])/S[a];
}


for(i in D2){
  W<-ridge_2(X_tr,y_tr,i)
  W0<-W[1];
  W<-W[-1];
  
  y_yu_tr<-vector();#预测+计算误差
  y_yu_te<-vector();
  
  for(a in 1:50){
    y_yu_tr[a]<-t(W)%*%X_tr[a,]+W0+My;
  }
  
  for(a in 1:47){
    y_yu_te[a]<-t(W)%*%X_te[a,]+W0+My;
  }
  
  cat('When the δ2 is',i,"The training error is:",sum((y_yu_tr-y_tr_o)^2)/length(X_tr),'\n')
  cat('When the δ2 is',i,"The testing error is:",sum((y_yu_te-y_te_o)^2)/length(X_te),'\n')
  
}


#=========================================================
#2.3_3
#寻找合适的δ2
#使用留一交叉验证法
#分开参数和结果
yy<-Shuju[,9]
XX<-Shuju[,-9]

#计算CV

CV<-vector()
for(i in 1:length(D2)){
  tem<-0;
  for(k in 1:97){
    #分开训练集
    XX_te<-XX[k,];
    yy_te<-yy[k];
    k<-k*(-1);
    XX_tr<-XX[k,];
    yy_tr<-yy[k];
    
    #训练集standardization
    S<-vector()    #标准差集
    M<-vector()    #均值集
    for(a in 1:8){   #对元素进行standerdise
      S[a]<-sd(XX_tr[,a]);
      M[a]<-mean(XX_tr[,a]);
      XX_tr[,a]<-(XX_tr[,a]-M[a])/S[a];
    }
    
    Myy<-mean(yy_tr);
    yy_tr<-(yy_tr-Myy);
    #测试集standardization
    for(a in 1:8){
      XX_te[a]<-(XX_te[a]-M[a])/S[a];
    }
    
    #预测并计算误差
    #训练
    W<-ridge_2(XX_tr,yy_tr,D2[i])
    W0<-W[1];
    W<-W[-1];
    
    #预测
    yy_yu_te<-t(W)%*%XX_te+W0+Myy;
    
    tem<-tem+(yy_yu_te-yy_te)^2;
    
  }
  CV[i]<-tem/97;
}

#作图
plot(xlim=c(-2,4),D2_p,CV,type="o",col=2,pch=20,cex=1,xlab='δ2',ylab="CV",xaxt="n",xaxs="i")
grid(nx=6,ny=5,lwd=1,lty=2,col="blue") 
axis(side=1,at = c(-2,-1,0,1,2,3,4),labels = c('10^-2','10^-1','10^0','10','10^2','10^3','10^4'))   
#找到最合适的δ2
cat('最合适的δ2为：',D2[which.min(CV)])


#对于不同的δ2求train\Testerror并绘图---------------------------------

Tr_err<-vector();
Te_err<-vector();

for(k in 1:length(D2)){       #求训练集与测试集的error
  W<-ridge_2(X_tr,y_tr,D2[k])
  W0<-W[1];
  W<-W[-1];
  
  y_yu_tr<-vector();#预测+计算误差
  y_yu_te<-vector();
  
  for(a in 1:50){
    y_yu_tr[a]<-t(W)%*%X_tr[a,]+W0+My;
  }
  
  for(a in 1:47){
    y_yu_te[a]<-t(W)%*%X_te[a,]+W0+My;
  }
  
  Tr_err[k]<-sqrt(sum((y_yu_tr-y_tr-My)^2))/sqrt(sum((y_tr+My)^2));
  Te_err[k]<-sqrt(sum((y_yu_te-y_te)^2))/sqrt(sum(y_te^2));
  
}

#作图
with(iris, plot(ylim=c(0,0.5),xlim=c(-2,4),D2_p,Te_err,type="o",col=2,pch=20,cex=1,xlab="δ2",ylab="||y-Xθ||2/||y||2",xaxt="n",xaxs="i",yaxs="i"))
grid(nx=6,ny=5,lwd=1,lty=2,col="blue") 

lines(D2_p,Tr_err,col=4,type='o',pch=20)
axis(side=1,at = c(-2,-1,0,1,2,3,4),labels = c('10^-2','10^-1','10^0','10','10^2','10^3','10^4'))   

#图例
legend(x=-2,y=0.5,c('Test_error','Train_error'),col=c(2,4),lty=1)
