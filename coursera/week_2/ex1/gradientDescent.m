function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
  %   taking num_iters gradient steps with learning rate alpha

  m = length(y);
  J_history = zeros(num_iters, 1);
  feature_count = size(X,2);

  for iter = 1:num_iters
    delta = zeros(feature_count, 1);

    for i = 1:m
      hypothesis = X(i,:) * theta;
      real_value = y(i);

      for feature_i = 1:feature_count
        delta(feature_i) += (hypothesis - real_value) * X(i, feature_i);
      end
    end

    J_history(iter) = computeCost(X, y, theta);
    theta = theta - alpha * (1 / m) * delta;
  end
end
