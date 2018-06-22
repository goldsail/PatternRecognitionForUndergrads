rng(2018);

% normal samples
x = [-3 : 0.01 : 3];

for N = [10, 100, 1000]
    
    h = figure();
    colors = ['r', 'b', 'g'];
    
    for j = 1:3
        X = randn(1, N);
        mu = mean(X);
        sigsq = (N-1)/N * var(X);
        
        y = exp(-1/2 * ((x - mu).^2) / sigsq) ./ (sqrt(2 * pi * sigsq));
        hold on;
        plot(x, y, colors(j));
        hold off;
    end
    
    y = exp(-1/2 * (x.^2)) ./ (sqrt(2 * pi));
    hold on;
    plot(x, y, 'k:');
    hold off;
    
    title(sprintf('Normal samples, N = %d', N));
    saveas(h, sprintf('normal_%d.png', N), 'png');
end

% uniform samples
x = [-2 : 0.01 : 3];

N = 100;
h = figure();
colors = ['r', 'b', 'g'];

for j = 1:3
    X = rand(1, N);
    mu = mean(X);
    sigsq = (N-1)/N * var(X);

    y = exp(-1/2 * ((x - mu).^2) / sigsq) ./ (sqrt(2 * pi * sigsq));
    hold on;
    plot(x, y, colors(j));
    hold off;
end

y = double(x > 0) .* double(x <= 1);
hold on;
plot(x, y, 'k:');
hold off;

title(sprintf('Uniform samples, N = %d', N));
saveas(h, sprintf('uniform_%d.png', N), 'png');
