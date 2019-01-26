function l = logdet(M,errorDet)

[R,p] = chol(M);  %Cholskey LU分解
if p ~= 0   %M 不是对称正定矩阵
   l = errorDet;
else
   l = 2*sum(log(diag(R)));
end;

