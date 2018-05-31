% TODO vectorize!

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
K = max(y);     % number of classes for the problem
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

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



% Add the column of 1s to X (they will be the value of the bias for each example)
X = [ones(m, 1), X];

for i = 1 : m  % loop over examples
  x_i = X(i,:)';   % input layer
  y_i       = zeros(K, 1);
  y_i(y(i)) = 1;
  % Layers 1 - 2
  a1 = x_i;
  z2 = Theta1*a1;
  a2 = sigmoid(z2); % WATCH OUT: do not add the unit bias before performing the sigmoid!
  a2 = [1; a2];     % HEEEYYYY!!
  % Layers 2 - 3
  z3 = Theta2*a2;
  h_i = sigmoid(z3);   % (10) hypotheses computed for the example i
  for k = 1 : K  % loop over the different classes
    y_ik = y_i(k);
    h_ik = h_i(k);
    J = J - ( y_ik*log(h_ik) + (1-y_ik)*log(1-h_ik) );
  end

  % Computing Backpropagation
  delta3 = h_i - y_i;
  z2 = [1; z2]; % add unit bias to the vector (also needed in computation of delta2)
  delta2 = (Theta2'*delta3).*sigmoidGradient(z2);
  % delta2 = Theta2'*(delta3.*sigmoidGradient(z3));
  % delta1 = Theta1'*(delta2.*sigmoidGradient(z2));    % Not needed - actually wrong!

  Theta1_grad = Theta1_grad + (delta2(2:end)*a1');
  Theta2_grad = Theta2_grad + (delta3*a2');
end

J = J/m;
Theta1_grad = Theta1_grad/m;
Theta2_grad = Theta2_grad/m;
% Regularize gradient (do not apply regularization to the first column!)
reg1 = lambda*Theta1(:,2:end)/m;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + reg1;
reg2 = lambda*Theta2(:,2:end)/m;
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + reg2;

% Compute regularization and sum it to the computed cost function
T1 = Theta1(:,2:end);
T2 = Theta2(:,2:end);
reg = lambda*( sum(sum(T1.*T1)) + sum(sum(T2.*T2)) )/(2*m);

J = J + reg;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
