function J = computeCost(X, y, theta)
  %COMPUTECOST Compute cost for linear regression
  %   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
  %   parameter for linear regression to fit the data points in X and y

  m = length(y);
  J = 0;

  for i = 1:m
    hypothesis = X(i,:) * theta;
    J += (hypothesis - y(i)) ** 2;
  end

  J /= (2 * m);
end
