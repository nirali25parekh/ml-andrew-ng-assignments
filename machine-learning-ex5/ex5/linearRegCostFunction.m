function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%----------cost----------------

%they have added x0 = 1 column for us
% X is 12x2, theta is 2x1, hx is 12x1
hx = X*theta;
% hx is 12x1, y is 12x1
costNoReg = (1/(2*m)) * sum((hx-y).^2);
% theta is 2x1, we have to remove theta0 in reg, add ip all squares of row
regTermCost = lambda/(2 * m) * sum(theta(2:end, :).^2);
J  = costNoReg + regTermCost;

%----------gradient------------

% hx is 12x1, y is 12x1, X' is 2x12,(hx-y) is 12x1,  prod part is 2x1
gradNoReg = (1/m) * X'*(hx-y) ; %gradNoReg is 2x1
regTermGrad = (lambda/m) * [0 ; theta(2:end, :)]; %regTermGrad is 2x1
grad = gradNoReg + regTermGrad;





% =========================================================================

grad = grad(:);

end
