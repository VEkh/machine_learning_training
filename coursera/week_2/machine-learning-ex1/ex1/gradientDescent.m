function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
  %GRADIENTDESCENT Performs gradient descent to learn theta
  %   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
  %   taking num_iters gradient steps with learning rate alpha

  % Initialize some useful values
  m = length(y); % number of training examples
  J_history = zeros(num_iters, 1);
  _theta = [theta];

  for iter = 1:num_iters
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    delta_0 = 0;
    delta_1 = 0;

    for i = 1:m
      prediction_difference = X(i,:) * _theta - y(i);
      delta_0 += prediction_difference * X(i, 1);
      delta_1 += prediction_difference * X(i, 2);
    end

    delta = [delta_0; delta_1] / m;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, _theta);
    _theta = _theta - alpha * delta;
  end

  theta = _theta;
end
