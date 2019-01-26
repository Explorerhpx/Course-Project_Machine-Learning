#1.2使用最大似然法进行拟合
Shuju<-read.table("C:\\Users\\DELL\\Desktop\\课件与课本\\3上\\统计（机器）学习\\pj\\project1\\data\\basicData.txt",header=1)
X_tr<-Shuju$X
y_tr<-Shuju$y
X_te<-Shuju$Xtest
y_te<-Shuju$Ytest

leastSquaresBasis<-function(x,y,deg){  #使用最大似然法
  Xpoly<-matrix(nrow=length(x),ncol=deg+1);
  for(a in 1:length(x)){
    for(b in 0:deg){
      Xpoly[a,(b+1)]<-(x[a])^b;
    }
  }
  W<-solve(t(Xpoly)%*%Xpoly)%*%t(Xpoly)%*%y;
  W<-W[,1];
  W_fuzhu<-vector();
  for(i in 1:(deg+1)){
    W_fuzhu[i]=mean(Xpoly[,i]);
  }
  W0<-mean(y)-t(W)%*%W_fuzhu;
  c(W0,W)
}


for(i in 0:8){ 
  W<-leastSquaresBasis(X_tr,y_tr,i) #regression
  W0<-W[1];
  W<-W[-1];
  
  Xpoly<-matrix(nrow=length(X_tr),ncol=i+1);
  for(a in 1:length(X_tr)){
    for(b in 0:i){
      Xpoly[a,(b+1)]<-(X_tr[a])^b;
    }
  }
  
  Xpoly_t<-matrix(nrow=length(X_te),ncol=i+1);
  for(a in 1:length(X_te)){
    for(b in 0:i){
      Xpoly_t[a,(b+1)]<-(X_te[a])^b;
    }
  }
  
  y_yu_tr<-vector();#预测+计算误差
  y_yu_te<-vector();
  
  for(a in 1:length(X_tr)){
    y_yu_tr[a]<-t(W)%*%Xpoly[a,]+W0;
  }
  
  for(a in 1:length(X_tr)){
    y_yu_te[a]<-t(W)%*%Xpoly_t[a,]+W0;
  }

  cat('When the deg is',i,"The training error is:",sum((y_yu_tr-y_tr)^2)/length(X_tr),'\n')
  cat('When the deg is',i,"The testing error is:",sum((y_yu_te-y_te)^2)/length(X_te),'\n')
}


