rng(2018);
sigma = 2;
N = 10;
Nt = 100;
theta1 = 3;
theta0 = 6;

% question 1
X = normrnd(0, 1, N, 1);
Y = theta1 * X + theta0 + normrnd(0, sigma^2, N, 1);

Xt = normrnd(0, 1, Nt, 1);
Yt = theta1 * Xt + theta0 + normrnd(0, sigma^2, Nt, 1);

% question 2 (linear)
disp('linear');
[B,BINT,R,RINT,STATS] = regress(Y, [ones(N, 1), X]);
disp(B);
disp(STATS(1));
h = figure();
scatter(X, Y);
X0 = (min(X) : 0.1 : max(X))';
Y0 = [ones(length(X0), 1), X0] * B;
hold on;
plot(X0, Y0);
hold off;
saveas(h, sprintf('2-2-linear-%d-%2.1f.png', N, sigma));

% question 2 (quadratic)
disp('quadratic');
[B,BINT,R,RINT,STATS] = regress(Y, [ones(N, 1), X, X.^2]);
disp(B);
disp(STATS(1));
h = figure();
scatter(X, Y);
X0 = (min(X) : 0.1 : max(X))';
Y0 = [ones(length(X0), 1), X0, X0.^2] * B;
hold on;
plot(X0, Y0);
hold off;
saveas(h, sprintf('2-2-quadratic-%d-%2.1f.png', N, sigma));

% question 2 (cubic)
disp('cubic');
[B,BINT,R,RINT,STATS] = regress(Y, [ones(N, 1), X, X.^2, X.^3]);
disp(B);
disp(STATS(1));
h = figure();
scatter(X, Y);
X0 = (min(X) : 0.1 : max(X))';
Y0 = [ones(length(X0), 1), X0, X0.^2, X0.^3] * B;
hold on;
plot(X0, Y0);
hold off;
saveas(h, sprintf('2-2-cubic-%d-%2.1f.png', N, sigma));

% question 3 (linear)
disp('test: linear');
[B,BINT,R,RINT,STATS] = regress(Y, [ones(N, 1), X]);
Yh = [ones(Nt, 1), Xt] * B;
disp(sumsqr(Yh - Yt));
disp(sumsqr(Yh - Yt) / sumsqr(Yt));

% question 3 (quadratic)
disp('test: quadratic');
[B,BINT,R,RINT,STATS] = regress(Y, [ones(N, 1), X, X.^2]);
Yh = [ones(Nt, 1), Xt, Xt.^2] * B;
disp(sumsqr(Yh - Yt));
disp(sumsqr(Yh - Yt) / sumsqr(Yt));

% question 3 (linear)
disp('test: cubic');
[B,BINT,R,RINT,STATS] = regress(Y, [ones(N, 1), X, X.^2, X.^3]);
Yh = [ones(Nt, 1), Xt, Xt.^2, Xt.^3] * B;
disp(sumsqr(Yh - Yt));
disp(sumsqr(Yh - Yt) / sumsqr(Yt));



