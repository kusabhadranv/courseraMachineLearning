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
idMat = ones(m,1);
X = [idMat X];
a1 = X;
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m,1) a2];
z3 = a2 * Theta2';
a3 = sigmoid(z3);
cost=0;
sa2 = size(a2);


for i = 1:m    
    label = zeros(num_labels,1);      %This matrix is to convert the output label into a binary matrix
    label(y(i)) = 1;                  %Make only the index corresponding to the value in y as 1
    hyp = a3(i,:)';                   %Get the hypothesis for example i
    cost = cost + sum((-1 * label .* log(hyp)) - ((1 - label) .* log(1 - hyp)));
            
end
J = cost/m;
 
%regTheta1 = sum(sum((Theta1(:,[2:end])) .^ 2));
regTheta1 = sum(((Theta1(:,[2:end])) .^ 2),'all');

%regTheta2 = sum(sum((Theta2(:,[2:end])) .^ 2));
regTheta2 = sum(((Theta2(:,[2:end])) .^ 2),'all');

regterm = (lambda / (2 * m)) * (regTheta1 + regTheta2);

J = J + regterm;

for t = 1:m
   %Step1 
   a1 = X(t,:);         %(1x401)
   z2 = a1 * Theta1';   %(1x401) * (401x25) = 1x25
   a2 = [1 sigmoid(z2)];%(1x26)
   z3 = a2 * Theta2';   %(1x26) * (26x10)
   a3 = sigmoid(z3);    %(1x10)
   
   %Step2
   label = zeros(num_labels,1);     
   label(y(t)) = 1;
   delta3 = a3' - label; %(10x1)
   
   %Step3
   z2 = [1 z2];
   delta2 = (Theta2' * delta3)' .* sigmoidGradient(z2); %(26x10) * (10x1) -> (26x1) .* (26x1)
   
   %Step4
   delta2 = delta2(2:end); %
   Theta2_grad = Theta2_grad + (delta3 * a2);
   Theta1_grad = Theta1_grad + (delta2' * a1);
   
   
   
end

% Step 5
Theta2_grad = (1/m) * Theta2_grad; 
Theta1_grad = (1/m) * Theta1_grad; 
regTheta2 = (lambda/m) * Theta2(:,[2:end]);
regTheta1 = (lambda/m) * Theta1(:,[2:end]);

Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + regTheta2;
Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + regTheta1;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
