
%% Define dataset

rng('default'); % Sets seed to 0

x = linspace(0, 2)';

% Generate noise
noise = 2 * randn(size(x));

% Generate regression equation
y = 3 * x + 4 * x .^ 2 + x .^3 + noise;

%% Test train split

N = length(x);

% Decide split
partition = cvpartition(N, 'HoldOut', 0.3);

% Finds location of data points used in dataset
test_indexes = partition.test;

% Identify training dataset by negating test indexes
x_train_initial = x(~test_indexes);
y_train = y(~test_indexes);

% Identify testing dataset
x_test_initial = x(test_indexes);
y_test = y(test_indexes);

% Prepare matrix for x-values
x_train = [ones(size(x_train_initial)), x_train_initial];
x_test = [ones(size(x_test_initial)), x_test_initial];

%% Figure out coeffiecient

% W = X_transpose * y
B = x_train \ y_train;

%% Make predictions on training dataset

y_train_predictions = x_train * B;

%% Identify how good the model is

mean_squared_error = mean( (y_train - y_train_predictions) .^ 2);
root_mean_squared_error = sqrt(mean_squared_error);

%% Make predictions on testing dataset

y_test_predictions = x_test * B;

%% Identify how good the model really is

mean_squared_error_test = mean( (y_test - y_test_predictions) .^ 2);
root_mean_squared_error_test = sqrt(mean_squared_error_test);

%% Print results

disp('y = b0 + b1 * x')
fprintf('b0: %f, \nb1: %f\n', B(1),B(2));
fprintf('RMSE(train): %f, and \nRMSE(test): %f\n', root_mean_squared_error, root_mean_squared_error_test);

%% Plot data

figure(1);clf

scatter(x,y)
hold on

plot(x_train, y_train_predictions)

title('y = b_0 + b_1 x')
xlabel('Independent variable, x')
ylabel('Dependent variable, y')

axis on
box on

hold off