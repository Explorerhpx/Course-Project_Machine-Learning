# 使用最小二乘法
Shuju<-read.table("C:\\Users\\DELL\\Desktop\\课件与课本\\3上\\统计（机器）学习\\pj\\project1\\data\\basicData.txt",header=1)
X_tr<-Shuju$X
y_tr<-Shuju$y
X_te<-Shuju$Xtest
y_te<-Shuju$Ytest

#最小二乘函数及辅助函数

leastSquaresBias<-function(X,Y){  #最小二乘法
  xm<-mean(X);
  ym<-mean(Y);
  P1<-sum(t((X-xm))*(Y-ym))/sum(t((X-xm))*(X-xm));
  P0<-ym-P1*xm;
  c(P1,P0);
}

P1<-leastSquaresBias(X_tr,y_tr)[1]
P0<-leastSquaresBias(X_tr,y_tr)[2]

Yuce<-function(X){   #最终预测函数
  X*P1+P0;
}

y_yu_tr<-Yuce(X_tr)
y_yu_te<-Yuce(X_te)

#绘图
plot(X_tr,y_tr,xlim=c(-10,10),ylim=c(-300,400),
     type="p",col=1,,cex=1, yaxs="i",xaxs="i",
     xlab="x",ylab="y")
lines(X_tr,y_yu_tr,col=4)

#training error/test error
paste("The training error is:",sum((y_yu_tr-y_tr)^2)/length(X_tr))
paste("The testing error is:",sum((y_yu_te-y_te)^2)/length(X_te))
