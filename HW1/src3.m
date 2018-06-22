X = load('X.txt');
Y = load('Y.txt');
N = length(Y);
X0 = [110, 3, 1];

% no interactive terms
X1 = [ones(N, 1), X];
[B1, BINT, R, RINT, STATS] = regress(Y, X1);
disp(B1);
disp(STATS(1));
Yh1 = [1, X0] * B1;
disp(Yh1);

% full interactive terms
X2 = [ones(N, 1), X, X(:,1).*X(:,2), X(:,2).*X(:,3), X(:,3).*X(:,1), ];
[B2, BINT, R, RINT, STATS] = regress(Y, X2);
disp(B2);
disp(STATS(1));
Yh2 = [1, X0, X0(:,1).*X0(:,2), X0(:,2).*X0(:,3), X0(:,3).*X0(:,1)] * B2;
disp(Yh2);

% half interactive terms
X3 = [ones(N, 1), X, X(:,2).*X(:,3), X(:,3).*X(:,1), ];
[B3, BINT, R, RINT, STATS] = regress(Y, X3);
disp(B3);
disp(STATS(1));
Yh3 = [1, X0, X0(:,2).*X0(:,3), X0(:,3).*X0(:,1)] * B3;
disp(Yh3);