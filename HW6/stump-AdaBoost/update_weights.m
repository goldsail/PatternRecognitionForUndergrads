function w_update = update_weights(X, y, j, a, d, w, c)
% ����Ȩֵ
% 
% ����
%     X        : n * p ����ÿһ����һ��ѵ������
%     y        : n * 1 ������ÿһ����һ��ѵ����ǩ
%     j        : ��ѡ�������ά��
%     a        : ��ѡ��ֵ
%     d        : 1 �� -1
%     w        : n * 1 ����, ԭ����Ȩֵ
%     c        : ������������Ȩֵ
%
% Output
%     w_update : n * 1 ����, ���¹����Ȩֵ

%%% �벹ȫ���� %%%


p = ((X(:, j) <= a) - 0.5) * 2 * d; % prediction labels
w_update = w .* exp(c * (p ~= y)); % update the weights

w_update = w_update / sum(w_update); % normalization


%%% �벹ȫ���� %%%
end