# ʹ����С���˷�
Shuju<-read.table("C:\\Users\\DELL\\Desktop\\�μ���α�\\3��\\ͳ�ƣ�������ѧϰ\\pj\\project1\\data\\basicData.txt",header=1)
X_tr<-Shuju$X
y_tr<-Shuju$y
X_te<-Shuju$Xtest
y_te<-Shuju$Ytest

#��С���˺�������������

leastSquaresBias<-function(X,Y){  #��С���˷�
  xm<-mean(X);
  ym<-mean(Y);
  P1<-sum(t((X-xm))*(Y-ym))/sum(t((X-xm))*(X-xm));
  P0<-ym-P1*xm;
  c(P1,P0);
}

P1<-leastSquaresBias(X_tr,y_tr)[1]
P0<-leastSquaresBias(X_tr,y_tr)[2]

Yuce<-function(X){   #����Ԥ�⺯��
  X*P1+P0;
}

y_yu_tr<-Yuce(X_tr)
y_yu_te<-Yuce(X_te)

#��ͼ
plot(X_tr,y_tr,xlim=c(-10,10),ylim=c(-300,400),
     type="p",col=1,,cex=1, yaxs="i",xaxs="i",
     xlab="x",ylab="y")
lines(X_tr,y_yu_tr,col=4)

#training error/test error
paste("The training error is:",sum((y_yu_tr-y_tr)^2)/length(X_tr))
paste("The testing error is:",sum((y_yu_te-y_te)^2)/length(X_te))