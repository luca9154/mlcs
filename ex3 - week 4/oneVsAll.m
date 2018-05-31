function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logisitc regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell use 
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%


% Notes about the trick used for the mutli-classification
% =====
% fmincg returns a column vector of weights trained on X and y.

% training can also be said "computing the weights"

% In a multiclass classification my vector y probably contains
% integers from 1 to num_labels, that represent the labels for
% the classification.

% In order to obtain the weights for the hypothesis number c,
% I must set y to 0 in each position but c, where I must set 1
% ( y = 0; y(c) = 1;).
% This is exactly equal to consider positive (1) the elements of the set
% which are within the class c and negative
% (0) all of the other data. This becomes then, again, a simple logistic
% regression on only 2 classes, corresponding to examples belonging to the
% class c and to all of the other examples not belonging to the class c
% (as seen in the previous week lectures).
% If I do this (setting to 0s all of the elements of y, and setting the element
% c to 1, before training) for each class, I get
% num_labels sets of thetas (the rows of all_theta), each of which I will then use as
% the corresponding hypothesis (the c-th hypothesis), i.e. the hypothesis that a new set
% of data (a new set of features, i.e. n values (x0=1), like a row of X) will belong
% to the class c.
% Doing that for each set of thetas (i.e. for each class) will give the probability for the data
% to belong to each of the classes: the set of data will finally be considered
% to belong to the class with the highest probability (i.e the index, in the probability
% array, of its biggest element) - see the implementation of predictOneVsAll function.

initial_theta = zeros(n + 1, 1);
options = optimset('GradObj', 'on', 'MaxIter', 50);
for c = 1:num_labels
  [theta] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
                    initial_theta, options);
  all_theta(c,:) = theta';
end






% =========================================================================


end
