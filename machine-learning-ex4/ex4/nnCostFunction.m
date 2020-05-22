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
    a1 = X;   % a1 is 5000 x 400
    a1WithBias = [ones(m,1) a1]; % we need 5000 x 401, so, column added on left
 
    z2 = a1WithBias * Theta1';   % a1WithBias is 5000 x 401 , Theta1 is 25 x 401, z2 = 5000 x 25
    a2 = sigmoid(z2); % a2 is 5000 x 25
    
    a2WithBias = [ones(m, 1) a2]; % we need 5000 x 26
    z3 = a2WithBias * Theta2';   % a2 is 5000 x 26, Theta2 is 10 x 26, z3 = 5000 x 10
    
    a3 = sigmoid(z3);  % a3 is 5000 x 10
    hx = a3; % hx is 5000 x 10 
   
    y2 = [];
    % to convery each 5000 y's into vector form, y is 5000x 1
    for i = 1:m,
     y2 = [y2; ([1:num_labels] == y(i))]; % append row to y2 where value 1 is correct class
    endfor;
 
    % y2 is 5000 x 10, log(hx) is 5000 x 10
    JNoReg = (-1 / m) * sum(sum(y2.*log(hx) + (1 - y2).*log(1 - hx))); %cost without regularization
    
%-------- start reg-------------------
    
    Theta1ZeroBias = [ zeros(size(Theta1, 1), 1) Theta1(: , 2:end) ];
    Theta2ZeroBias = [ zeros(size(Theta2, 1), 1) Theta2(: , 2:end) ];
    
    regTerm = (lambda / (2 * m))*(sum(sum((Theta1ZeroBias.^2)))+sum(sum((Theta2ZeroBias.^2)))) ;
    
    J = JNoReg + regTerm;

%----------start gradient - non reg--------  

Delta1 = zeros(size(Theta1)); %25x401
Delta2 = zeros(size(Theta2));   %10x26

for t = 1:m 
  
     a1tWithBias = a1WithBias(t, :);  % 1x401
     z2t = z2(t,:);   %1 x 25
     a2tWithBias = a2WithBias(t, :);   % 1x26
     a3t = a3(t, :);   %1x10
     yt = y2(t, :);    %1x10
     
    err3 = a3t - yt; % a3t is 1 x 10, yt is 1 x 10, err3 is 1 x 10
    
    err2 = (err3 * Theta2) .* sigmoidGradient([1 z2t]);
            %1x10 * 10x26 = 1x26 .* 1x26 =  1x26
    Delta1 = Delta1 + err2(2:end)'*a1tWithBias; % err2 is 1 x 25, a1 is 1 x 401, their prod is 25x401
    Delta2 = Delta2 + err3'*a2tWithBias; % err3 is 1 x 10, a2 is 1x26, their prod is 10x26
    
endfor

    Theta1_grad = (1/m) * Delta1 + (lambda / m) * Theta1ZeroBias;
    Theta2_grad = (1/m) * Delta2 + (lambda / m) * Theta2ZeroBias;
% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
