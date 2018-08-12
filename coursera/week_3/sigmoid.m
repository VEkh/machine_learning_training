function g = sigmoid(z)
  %SIGMOID Compute sigmoid function
  %   g = SIGMOID(z) computes the sigmoid of z.

  % You need to return the following variables correctly 
  dimensions = size(z);
  g = zeros(dimensions);

  % ====================== YOUR CODE HERE ======================
  % Instructions: Compute the sigmoid of each value of z (z can be a matrix,
  %               vector or scalar).

  for row = 1:dimensions(1)
    for column = 1:dimensions(2)
      g(row, column) = 1 / (1 + e ** -z(row, column));
    end
  end
end
