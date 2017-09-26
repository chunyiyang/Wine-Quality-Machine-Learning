clear all; close all; clc;

  data = csvread("winequality-red.csv",1,0);
  normalize = max(data)-min(data);

  % normalize 
  data(:,1:11) = data(:,1:11)./normalize(1:11);
 
  K = 10;
  index = randperm(length(data));          
  data_shuffled = data(index, :);
  data_shuffled = data_shuffled(1:1500, :);
  data_shuffled = [repmat(1,length(data_shuffled),1) data_shuffled]; % = [1 X]
  [nums, features] = size(data_shuffled);
  accuracy = zeros(K,1);
  
  for k=1:K
 
    test_index = [1+(k-1)*nums/10:k*nums/10];
    train_index = [1:(k-1)*nums/10,k*nums/10+1:nums];
  
    train_data = data_shuffled(train_index,:);
    
    test_data = data_shuffled(test_index,1:12);
    test_label = data_shuffled(test_index,13);

    w = zeros(12,1); % size w is 2x1
    w =inv(transpose(train_data(:,1:12)) * train_data(:,1:12)) * transpose(train_data(:,1:12)) * train_data(:,13);

    yhat = (test_data*w);
    
    testnums = length(test_data);
    for i=1:testnums
      if (round(yhat(i)) == test_label(i))
        accuracy(k,1) += 1;
      endif
    end
    accuracy(k,1) = accuracy(k,1)/testnums;
  end
  
  fprintf("Classification accuracy\n");
  fprintf("ans = \n");
  disp(accuracy)
  fprintf('average = %5.4f \n\n', sum(accuracy(:,1))/K);
