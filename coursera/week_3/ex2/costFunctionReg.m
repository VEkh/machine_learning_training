function [J, grad] = costFunctionReg(theta, X, y, lambda)
  %COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
  %   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
  %   theta as the parameter for regularized logistic regression and the
  %   gradient of the cost w.r.t. to the parameters.

  m = length(y);
  J = 0;
  theta_length = size(theta, 1);
  grad = zeros(theta_length, 1);

  [J, grad] = costFunction(theta, X, y);

  for i = 2:(theta_length)
    J += (lambda / (2 * m)) * theta(i) ** 2;
    grad(i) += (lambda / m) * theta(i);
  end
end
