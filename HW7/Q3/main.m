X = load('feature_selection_X.txt');
Y = load('feature_selection_Y.txt');

dat_full = [Y, X];

P = size(X, 2);
N = size(X, 1);
p = 20;

divergence = zeros(1, P);
for i = 1:P
    d0 = ksdensity(X(Y==0, i));
    d1 = ksdensity(X(Y==1, i));
    
    divergence(i) = sum((d0 - d1) .* log(d0 ./ d1));
    
end

[~, i] = sort(divergence, 'descend');
d = i(1:p);
disp(sprintf('choose %d features out of %d. They are:', p, P));
disp(d');
dat_red = [Y, X(:, d)];

[classifierFull, accuracyFull] = trainClassifierFull(dat_full);
[classifierReduced, accuracyReduced] = trainClassifierReduced(dat_red);

disp(sprintf('full accuracy: %f', accuracyFull));
disp(sprintf('reduced accuracy: %f', accuracyReduced));
