function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
m   = size(X,1);
idx = zeros(m, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%


% loop over all the examples
for i = 1 : m
  x = X(i,:); % row vector
  smallest_dist = 100000;  % this is not optimal, but may work
  % loop over all the centroids
  for c = 1 : K
    c_c = centroids(c, :); % row vector
    diff_x_c = x - c_c;
    distance = sqrt(diff_x_c*diff_x_c');
    if (distance < smallest_dist)
      idx(i) = c;
      smallest_dist = distance;
    end
  end
end


% =============================================================

end

