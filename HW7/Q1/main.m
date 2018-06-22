D = xlsread('Q1.xlsx', 'Sheet1', 'B2:I9');
D = D.^2;
N = size(D, 1);

J = eye(N, N) - ones(N, N) / N;

B = - 1/2 * J * D * J;

[V, D] = eig(B);
[D, I] = sort(diag(D), 'descend');
V = V(:, I);
D = diag(D);

X = sqrt(D(1:2, 1:2)) * V(:, 1:2)';

h = figure('rend', 'painters', 'pos', [10 10 900 600]);
scatter(X(1, :), X(2, :));
title('MDS result (raw)');
labels = cellstr(["Wuhan", "Zhengzhou", "Beijing", "Zhoukou", ...
    "Yuncheng", "Shiyan", "Hanzhong", "Chongqing"]);
dx = 0.2; dy = 0.2; text(X(1, :) + dx, X(2, :) + dy, labels);
saveas(h, 'result', 'png')

a = 150 / 180 * pi;
Y = [cos(a), sin(a); -sin(a), cos(a)] * [-1, 0; 0, 1] * X;

h = figure('rend', 'painters', 'pos', [10 10 900 600]);
scatter(Y(1, :), Y(2, :));
title('MDS result (rotated)');
labels = cellstr(["Wuhan", "Zhengzhou", "Beijing", "Zhoukou", ...
    "Yuncheng", "Shiyan", "Hanzhong", "Chongqing"]);
dx = 0.2; dy = 0.2; text(Y(1, :) + dx, Y(2, :) + dy, labels);
saveas(h, 'rotated', 'png')