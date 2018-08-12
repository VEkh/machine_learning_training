function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
  %NNCOSTFUNCTION Implements the neural network cost function for a two layer
  %neural network which performs classification
  %   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
  %   X, y, lambda) computes the cost and gradient of the neural network. The
  %   parameters for the neural network are "unrolled" into the vector
  %   nn_params and need to be converted back into the weight matrices.
  %
  %   The returned parameter grad should be a "unrolled" vector of the
  %   partial derivatives of the neural network.
  %

  % Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
  % for our 2 layer neural network
  Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                   hidden_layer_size, (input_layer_size + 1));

  Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                   num_labels, (hidden_layer_size + 1));

  % Setup some useful variables
  m = size(X, 1);

  % You need to return the following variables correctly
  J = 0;

  % ====================== YOUR CODE HERE ======================
  % Instructions: You should complete the code by working through the
  %               following parts.
  %
  % Part 1: Feedforward the neural network and return the cost in the
  %         variable J. After implementing Part 1, you can verify that your
  %         cost function computation is correct by verifying the cost
  %         computed in ex4.m
  %
  % Part 2: Implement the backpropagation algorithm to compute the gradients
  %         Theta1_grad and Theta2_grad. You should return the partial derivatives of
  %         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
  %         Theta2_grad, respectively. After implementing Part 2, you can check
  %         that your implementation is correct by running checkNNGradients
  %
  %         Note: The vector y passed into the function is a vector of labels
  %               containing values from 1..K. You need to map this vector into a
  %               binary vector of 1's and 0's to be used with the neural network
  %               cost function.
  %
  %         Hint: We recommend implementing backpropagation using a for-loop
  %               over the training examples if you are implementing it for the
  %               first time.
  %
  % Part 3: Implement regularization with the cost function and gradients.
  %
  %         Hint: You can implement this around the code for
  %               backpropagation. That is, you can compute the gradients for
  %               the regularization separately and then add them to Theta1_grad
  %               and Theta2_grad from Part 2.
  %

  bias_vector = ones(m, 1);
  Z2 = [bias_vector X] * Theta1';
  A2 = sigmoid(Z2); % 5000x401 * 401x25 => 5000x25
  A3 = sigmoid([bias_vector A2] * Theta2'); % 5000x26 * 26x10 => 5000x10

  y2class_vector = @(y_t) [1:num_labels] == y_t;
  Class_y = cell2mat(arrayfun(y2class_vector, y, "UniformOutput", false));

  % for class = 1:num_labels
  %   class_y = y == class;
  %   hypothesis = A3(:, class);

  %   J += -class_y .* log(hypothesis) - (1 - class_y) .* log(1 - hypothesis);
  % end

  % Class_y (5000x10) .* A3(5000x10)
  J = -Class_y .* log(A3) - (1 - Class_y) .* log(1 - A3);
  J = mean(sum(J, 2));

  Reg_Theta1 = sum(Theta1(:, 2:end)(:) .** 2);
  Reg_Theta2 = sum(Theta2(:, 2:end)(:) .** 2);
  regularization_factor = (lambda / (2 * m)) * sum([Reg_Theta1 Reg_Theta2]);

  J += regularization_factor;

  % Gradients

  % Theta1_grad = zeros(size(Theta1));
  % Theta2_grad = zeros(size(Theta2));
  %
  % for t = 1:m
  %   % Theta1 = 25x401
  %   % Theta2 = 10x26

  %   x_t = X(t, :); % 1x400
  %   z2 = Theta1 * [1 x_t]'; % 25x401 * 401x1 => 25x1
  %   a2_t = sigmoid(z2); % 25x1

  %   % Probability of being in class
  %   a3_t = sigmoid(Theta2 * [1; a2_t]); % 10x26 * 26x1 => 10x1;

  %   y_t = [1:num_labels](:) == y(t);
  %   delta_3 = a3_t - y_t; % 10x1
  %   % 26x10 * 10x1 => 26x1
  %   delta_2 = Theta2' * delta_3 .* sigmoidGradient([1; z2]);

  %   % 25x1 * 1x401
  %   Theta1_grad += (1/m) * delta_2(2:end) * [1 x_t];
  %   % 10x1 * 1x26
  %   Theta2_grad += (1/m) * delta_3 * [1; a2_t]';
  % end

  Delta_3 = A3 - Class_y; % 5000x10
  Delta_2 = Delta_3 * Theta2 .* sigmoidGradient([bias_vector Z2]); % 5000x26

  Theta1_grad = (1/m) * Delta_2(:, 2:end)' * [bias_vector X]; % 25x5000 * 5000x401 => 25x401
  Theta2_grad = (1/m) * Delta_3' * [bias_vector A2]; % 10x5000 * 5000x26 => 10x26

  reg_Theta1_grad = Theta1_grad(:, 2:end) + (lambda / m) * Theta1(:, 2:end);
  reg_Theta2_grad = Theta2_grad(:, 2:end) + (lambda / m) * Theta2(:, 2:end);

  Theta1_grad = [Theta1_grad(:, 1) reg_Theta1_grad];
  Theta2_grad = [Theta2_grad(:, 1) reg_Theta2_grad];

  % Unroll gradients
  grad = [Theta1_grad(:) ; Theta2_grad(:)];
end
