function l = logdet(M,errorDet)

[R,p] = chol(M);  %Cholskey LU�ֽ�
if p ~= 0   %M ���ǶԳ���������
   l = errorDet;
else
   l = 2*sum(log(diag(R)));
end;

