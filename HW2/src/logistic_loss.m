function loss = logistic_loss( theta )

global X;
global Y;
global indeces_training;
global indeces_validation;

x = X(indeces_training, :);
y = Y(indeces_training, :);

h = 1 ./ (1 + exp(- x * theta));

loss = y .* log(h) + (1-y) .* log(1-h);
loss = sum(-loss);

end

