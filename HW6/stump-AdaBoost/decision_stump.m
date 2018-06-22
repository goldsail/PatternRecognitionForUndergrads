function [j, a, d] = decision_stump(X, y, w)
% 优化决策树桩的参数
% 
%
% 输入
%     X : n * p 矩阵, 每一行是一个样本
%     y : n * 1 向量, 每一行是一个标签
%     w : n * 1 向量, 权重
%
% 输出
%     j : 最优的维度
%     a : 最优的阈值
%     d : 最优的d，-1或者+1

% 请注意优化代码
%%% 请补全代码 %%%

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

%%% 请补全代码 %%%
end