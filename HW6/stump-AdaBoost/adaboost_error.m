function e = adaboost_error(X, y, j, a, d, c)
% adaboost_error: ����AdaBoost�������Ĵ�����
% 
% ����
%     X     : n * p ����,ÿһ����һ������
%     y     : n * 1 ������ÿһ����һ����ǩ
%     j     : M * 1 ����, ��ѡ������ά��
%     a     : M * 1 ����, ��ѡ����ֵ
%     d     : M * 1 ����, 1 �� -1
%     c     : M * 1 ����, ��������Ȩ��
%
% ���
%     e     : ������      

%%% �벹ȫ���� %%%

n = size(X, 1);
p = size(X, 2);
M = size(c, 1);

predictions = zeros(n, M); % predictions of all classifiers

m = find(j > 0, 1, 'last'); % the first m classifiers obtained

for i=1:m
    predictions(:,i) = ((X(:, j(i)) <= a(i)) - 0.5) * 2 * d(i); 
end

prediction = (((predictions * c) > 0) - 0.5) * 2; % take a vote
e = mean(prediction ~= y);

%%% �벹ȫ���� %%%
end