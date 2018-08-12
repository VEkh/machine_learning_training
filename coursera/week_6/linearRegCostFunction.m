function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
  %LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
  %regression with multiple variables
  %   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
  %   cost of using theta as the parameter for linear regression to fit the 
  %   data points in X and y. Returns the cost in J and the gradient in grad

  m = length(y);

  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the cost and gradient of regularized linear 
  %               regression for a particular choice of theta.
  %
  %               You should set J to the cost and grad to the gradient.
  %

  % X (12x2) * theta (2x1) => 12x1
  hypothesis = X * theta;

  J = sum((hypothesis - y) .^ 2);
  J_regularization_factor = lambda * sum(theta(2:end) .^ 2);
  J = (1 / (2 * m)) * (J + J_regularization_factor);

  % X' (2x12) * (hypothesis - y) (12x1) => 2x1
  grad = X' * (hypothesis - y);
  grad = (1 / m) * [grad(1); grad(2:end) + lambda * theta(2:end)];

  grad = grad(:);
end
