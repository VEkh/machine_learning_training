function [J, grad] = costFunction(theta, X, y)
  %COSTFUNCTION Compute cost and gradient for logistic regression
  %   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
  %   parameter for logistic regression and the gradient of the cost
  %   w.r.t. to the parameters.

  m = length(y);

  J = 0;
  theta_length = size(theta, 1);
  grad = zeros(theta_length, 1);

  for i = 1:m
    X_i = X(i, :);
    hypothesis_i = sigmoid(X_i * theta);
    y_i = y(i);

    J += (
      (1 / m) *
      (-y_i * log(hypothesis_i) - (1 - y_i) * log(1 - hypothesis_i))
    );

    for grad_i = 1:(theta_length)
      grad(grad_i) += (1 / m) * (hypothesis_i - y_i) * X_i(grad_i);
    end
  end
end
