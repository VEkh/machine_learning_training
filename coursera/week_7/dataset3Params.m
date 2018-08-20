function [C, sigma] = dataset3Params(X, y, Xval, yval)
  %DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
  %where you select the optimal (C, sigma) learning parameters to use for SVM
  %with RBF kernel
  %   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and
  %   sigma. You should complete this function to return the optimal C and
  %   sigma based on a cross-validation set.
  %

  % Values that were determined after running the below code to determine the
  % permutation with the smallest error.
  C = 1;
  sigma = 0.1;

  % ====================== YOUR CODE HERE ======================
  % Instructions: Fill in this function to return the optimal C and sigma
  %               learning parameters found using the cross validation set.
  %               You can use svmPredict to predict the labels on the cross
  %               validation set. For example,
  %                   predictions = svmPredict(model, Xval);
  %               will return the predictions on the cross validation set.
  %
  %  Note: You can compute the prediction error using
  %        mean(double(predictions ~= yval))
  %

  % values = [0.01 0.03 0.1 0.3 1 3 10 30];
  % errors = [];

  % for i = 1:length(values)
  %   for j = 1:length(values)
  %     c = values(i);
  %     s = values(j);

  %     model = svmTrain(X, y, c, @(x1, x2) gaussianKernel(x1, x2, s));
  %     predictions = svmPredict(model, Xval);
  %     err = mean(double(predictions != yval));

  %     errors = [errors; c s err];
  %   end
  % end

  % [_, min_indecies] = min(errors);
  % min_error_params = errors(min_indecies(3), 1:2);
  % C = min_error_params(1);
  % sigma = min_error_params(2);
end
