function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
n_examples = size(X,1);   % PV
p = zeros(n_examples, 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%


X = [ones(m,1) X];  % adding the bias vector (a 1 as first element of each example)

Xt = X';

for ex = 1:n_examples
  % first layer
  X_ex  = Xt(:,ex);        %  (401 x 1) the ex-th example
  z1_ex = Theta1*X_ex;     %  ( 25 x 1)
  a1_ex = sigmoid(z1_ex);  %  ( 25 x 1)

  % second layer
  a1_ex = [1; a1_ex];      %  ( 26 x 1)
  z2_ex = Theta2*a1_ex;    %  (num_labels x 1)
  a2_ex = sigmoid(z2_ex);  %  (num_labels x 1)

  [max_ex p(ex)] = max(a2_ex);
end


% =========================================================================


end
