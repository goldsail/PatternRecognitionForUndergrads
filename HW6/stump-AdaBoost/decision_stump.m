function [j, a, d] = decision_stump(X, y, w)
% �Ż�������׮�Ĳ���
% 
%
% ����
%     X : n * p ����, ÿһ����һ������
%     y : n * 1 ����, ÿһ����һ����ǩ
%     w : n * 1 ����, Ȩ��
%
% ���
%     j : ���ŵ�ά��
%     a : ���ŵ���ֵ
%     d : ���ŵ�d��-1����+1

% ��ע���Ż�����
%%% �벹ȫ���� %%%

n = size(X, 1);
p = size(X, 2);

A = zeros(1, p);
D = zeros(1, p);
R = zeros(1, p);

for i = 1:p
    x = unique(X(:, i));
    tempD = zeros(size(x));
    tempR = zeros(size(x));
    
    for k = 1:length(x)
        a = x(k);
        
        temp1 = decision_stump_error(X, y, i, a, 1, w);
        temp2 = decision_stump_error(X, y, i, a, -1, w);
        
        if temp1 < temp2
            tempD(k) = 1;
            tempR(k) = temp1;
        else
            tempD(k) = -1;
            tempR(k) = temp2;
        end
        
    end
    
    [R(i), k] = min(tempR);
    A(i) = x(k);
    D(i) = tempD(k);
    
end

[~, j] = min(R);

a = A(j);
d = D(j);

%%% �벹ȫ���� %%%
end