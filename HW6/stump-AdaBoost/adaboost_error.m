function e = adaboost_error(X, y, j, a, d, c)
% adaboost_error: 返回AdaBoost分类器的错误率
% 
% 输入
%     X     : n * p 矩阵,每一行是一个样本
%     y     : n * 1 向量，每一行是一个标签
%     j     : M * 1 向量, 所选的特征维度
%     a     : M * 1 向量, 所选的阈值
%     d     : M * 1 向量, 1 或 -1
%     c     : M * 1 向量, 分类器的权重
%
% 输出
%     e     : 错误率      

%%% 请补全代码 %%%

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

%%% 请补全代码 %%%
end