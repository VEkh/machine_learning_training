function idx = findClosestCentroids(X, centroids)
  %FINDCLOSESTCENTROIDS computes the centroid memberships for every example
  %   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
  %   in idx for a dataset X where each row is a single example. idx = m x 1
  %   vector of centroid assignments (i.e. each entry in range [1..K])
  %

  % Set K
  K = size(centroids, 1);

  % You need to return the following variables correctly.

  % ====================== YOUR CODE HERE ======================
  % Instructions: Go over every example, find its closest centroid, and store
  %               the index inside idx at the appropriate location.
  %               Concretely, idx(i) should contain the index of the centroid
  %               closest to example i. Hence, it should be a value in the
  %               range 1..K
  %
  % Note: You can use a for-loop over the examples to compute this.
  %

  % While this is clever and seems optimized, it's slower.
  % centroid_dist_fn = @(centroid_i, x_i)...
  %   norm(X(x_i, :) - centroids(centroid_i, :)) ** 2;

  % calc_distances_fn = @(i) arrayfun(centroid_dist_fn, 1:K, i);

  % distances = arrayfun(calc_distances_fn, 1:size(X,1), 'UniformOutput', false);

  % [_, idx] = min(cell2mat(distances), [], 2);

  distances = [];

  for i = 1:size(X, 1)
    x_i = X(i, :);

    for j = 1:K
      distances(i, j) = norm(x_i - centroids(j, :)) ** 2;
    end
  end

  [_, idx] = min(distances, [], 2);
end
