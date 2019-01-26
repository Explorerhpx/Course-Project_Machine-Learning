function d = l2_distance(a,b,df)  %d(x,y): a(x) �� b��y���ľ���
% L2_DISTANCE - computes Euclidean distance matrix
%
% E = L2_distance(A,B)
%
%    A - (DxM) matrix 
%    B - (DxN) matrix
%    df = 1, force diagonals to be zero; 0 (default), do not force
% 
% Returns:
%    E - (MxN) Euclidean distances between vectors in A and B
%
%
% Description : 
%    This fully vectorized (VERY FAST!) m-file computes the 
%    Euclidean distance between two vectors by:
%
%                 ||A-B|| = sqrt ( ||A||^2 + ||B||^2 - 2*A.B )
%
% Example : 
%    A = rand(400,100); B = rand(400,200);
%    d = distance(A,B);

% Author   : Roland Bunschoten
%            University of Amsterdam
%            Intelligent Autonomous Systems (IAS) group
%            Kruislaan 403  1098 SJ Amsterdam
%            tel.(+31)20-5257524
%            bunschot@wins.uva.nl
% Last Rev : Wed Oct 20 08:58:08 MET DST 1999
% Tested   : PC Matlab v5.2 and Solaris Matlab v5.3

% Copyright notice: You are free to modify, extend and distribute 
%    this code granted that the author of the original code is 
%    mentioned as the original author of the code.

% Fixed by JBT (3/18/00) to work for 1-dimensional vectors
% and to warn for imaginary numbers.  Also ensures that
% output is all real, and allows the option of forcing diagonals to
% be zero.  

if (nargin < 2)   %������������
   error('Not enough input arguments');
end

if (nargin < 3)   %Ĭ��df = 0
   df = 0;    % by default, do not force 0 on the diagonal
end

if (size(a,1) ~= size(b,1))    %�޷�����a,b�и�����֮��ľ���
   error('A and B should be of same dimensionality');
end

if ~(isreal(a)*isreal(b))   %��������
   disp('Warning: running distance.m with imaginary numbers.  Results may be off.'); 
end

if (size(a,1) == 1)   %һ������
  a = [a; zeros(1,size(a,2))];   %��չ����Ϊ�����������ڶ��ж���0
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = sqrt(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);  %������������ģֵ��ά�������Թ������

% make sure result is all real
d = real(d); 

% force 0 on the diagonal? 
if (df==1)     %ǿ�ȶԽ���Ԫ��Ϊ0,but why?
  d = d.*(1-eye(size(d)));
end
