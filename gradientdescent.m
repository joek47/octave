%Command line parameters
arg_list = argv ();

%Loading data
data_name = arg_list{1};


%Identify the model name
model_name = arg_list{2};

[X_train, Y_train, X_val, Y_val, X_test, Y_test] = data_split(data_name);

% to boolean
Y_train = Y_train == 1;
Y_val = Y_val == 1;
Y_test = Y_test == 1;

%Weight initialization
feature_size = size(X_train, 2);         %The first feature equals 1 for all instances.
weights = randn(feature_size, 1) * 0.05; %Bias is integrated in weights.

iterations = str2num(arg_list{3});                                    

if strcmp(model_name, "fminunc")
	%Training
    options = optimset('GradObj', 'on', 'MaxIter', iterations);
    [weights, loss_train] = fminunc(@(t)(costFunction(t, X_train, Y_train)), weights, options);

	accuracy_val = logisticR_predict(X_val, Y_val, weights);
	accuracy_test = logisticR_predict(X_test, Y_test, weights);

elseif strcmp(model_name, "batch")
	% iterations = 100;
    alpha = str2double(arg_list{4});
    loss_history = zeros(iterations, 2); %first col for training loss, second col for val loss
    accuracy_history = zeros(iterations, 1); 
    weights_best = weights;
    val_acc_best = 0;

	for i = 1:1:iterations
		%Training
		% [loss_train, weights] = logisticR_train(X_train, Y_train, weights,alpha);

        %gradient descent step
        [loss_train, grad_train] = costFunction(weights, X_train, Y_train);
        weights = weights - alpha * grad_train;
        loss_history(i,1) = loss_train;

		if i <= 10
		    printf("Iteration %d loss: %f\n", i, loss_train);
		end
		%Evauate on validation data set
		accuracy_val = logisticR_predict(X_val, Y_val, weights);
        accuracy_history(i) = accuracy_val;

        if accuracy_val > val_acc_best
            val_acc_best = accuracy_val;
            weights_best = weights;
        endif
	endfor

    if arg_list{5} == 'd'
        % Plot the convergence graph 
        filename = strcat('convergence-', arg_list{1}, '-', arg_list{2}, '-', arg_list{3}, '-', arg_list{4},'.png');   
        plot_convergence(loss_history,accuracy_history, filename);
    endif

	%Evaluate on testing data set
	accuracy_val = logisticR_predict(X_val, Y_val, weights_best);
	accuracy_test = logisticR_predict(X_test, Y_test, weights_best);
else
	printf("Training model should be provided!\n")
	return
end

if arg_list{5} == 'd'
    %printf("====X_train ====\n")
    %X_train(1:5,:)
    %printf("====printing weights ====\n")
    %weights(1:5)
    %keyboard;

    %get p-value

endif

printf("Final loss for training data: %f\n", loss_train);
printf("Final accuracy for validation data: %f\n", accuracy_val);
printf("Final accuracy for test data: %f\n", accuracy_test);
