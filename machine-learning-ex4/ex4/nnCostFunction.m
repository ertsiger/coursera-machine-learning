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
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m

% Build the new y matrix
Y = zeros(m, num_labels);

for i=1:m
   Y(i, y(i)) = 1; 
end

% Compute h0 (a3)
a1 = [ones(m,1), X];

z2 = a1 * Theta1';
a2 = [ones(m,1), sigmoid(z2)];

z3 = a2 * Theta2';
a3 = sigmoid(z3);

% Compute cost
cost_pos = -Y .* log(a3);
cost_neg = - (1 - Y) .* log(1 - a3);
cost = cost_pos + cost_neg;
J = (1 /m) * sum(cost(:));

% Implementation using bucles (bad practice)
% for i=1:m
%    for k=1:num_labels
%       J = J + -Y(i,k) * log(a3(i,k)) - (1 - Y(i,k)) * log(1 - a3(i,k)); 
%    end
% end
% 
% J = J / m;

% Adding regularization term
cols_theta1 = size(Theta1, 2);
cols_theta2 = size(Theta2, 2);

Theta1_prime = Theta1(:,2:cols_theta1);
Theta2_prime = Theta2(:,2:cols_theta2);

Theta1_sq = Theta1_prime .^ 2;
Theta2_sq = Theta2_prime .^ 2;

J = J + (lambda / (2 * m)) * (sum(Theta1_sq(:)) + sum(Theta2_sq(:)));

%%
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

Delta1 = 0;
Delta2 = 0;

for t=1:m
    % Step 1
    a1 = [1, X(t,:)];
    z2 = a1 * Theta1';
    a2 = [1, sigmoid(z2)];
    z3 = a2 * Theta2';
    a3 = sigmoid(z3);
    
    % Step 2
    yt = Y(t,:);
    d3 = a3 - yt;
    
    % Step 3
    d2 = (d3 * Theta2_prime) .* sigmoidGradient(z2);
    %d2 = d2(2:end);
    
    % Step 4
    Delta1 = Delta1 + d2' * a1;
    Delta2 = Delta2 + d3' * a2;
end

% Step 5
Theta1_grad = (1 / m) * Delta1;
Theta2_grad = (1 / m) * Delta2;

%%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

Theta1_grad(:,2:end) = Theta1_grad(:, 2:end) + (lambda / m) * Theta1_prime;
Theta2_grad(:,2:end) = Theta2_grad(:, 2:end) + (lambda / m) * Theta2_prime;

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
