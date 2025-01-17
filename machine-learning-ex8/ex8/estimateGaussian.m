function [mu sigma2] = estimateGaussian(X)
%ESTIMATEGAUSSIAN This function estimates the parameters of a 
%Gaussian distribution using the data in X
%   [mu sigma2] = estimateGaussian(X), 
%   The input X is the dataset with each n-dimensional data point in one row
%   The output is an n-dimensional vector mu, the mean of the data set
%   and the variances sigma^2, an n x 1 vector
% 

% Useful variables
[m, n] = size(X);

% You should return these values correctly
mu = zeros(n, 1);
sigma2 = zeros(n, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the mean of the data and the variances
%               In particular, mu(i) should contain the mean of
%               the data for the i-th feature and sigma2(i)
%               should contain variance of the i-th feature.
%
% m = 37, n = 2
total = (sum(X))'; % total is 2x1
mu = total./m; % mu is 2x1

sigma2Mat = zeros(m, n);

for j = 1:m
  sigma2Mat(j,:) = ((X(j,:)) - (mu')).^2;  % each sigma2Mat is 1x2
endfor

sigma2 = (1/m)*(sum(sigma2Mat))';










% =============================================================


end
