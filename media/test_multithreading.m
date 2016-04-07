clear all
close all

% learning rate = 0.5
% weight decay = 5
% 400 hidden layers

% monothreaded
Ym1 = [96.08, 96.83, 97.19, 97.76, 97.43, ...
	   97.62, 97.61, 97.52, 97.80, 97.85, ...
	   97.52, 97.90, 97.94, 97.82, 98.24];
Ym2 = [96.10, 96.96, 97.42, 97.53, 97.46, ...
       97.31, 97.42, 97.44, 97.54, 97.87, ...
       97.77, 97.88, 97.50, 97.98, 97.51];
Ym3 = [96.28, 96.88, 97.42, 97.26, 97.19, ...
	   97.63, 97.69, 97.77, 97.83, 97.86, ...
	   97.83, 98.12, 97.67, 98.08, 97.77];

% multithreaded (4 threads)
Yq1 = [95.66, 96.88, 97.11, 97.26, 97.12, ...
	   97.04, 97.39, 97.59, 97.10, 97.59, ...
	   97.66, 97.72, 97.55, 97.84, 97.59];
Yq2 = [96.04, 96.73, 96.64, 96.71, 97.51, ...
       97.62, 97.79, 97.29, 97.45, 97.51, ...
       97.57, 97.73, 97.56, 97.75, 97.53];
Yq3 = [95.78, 96.87, 97.23, 97.43, 97.41, ...
	   97.39, 97.52, 97.17, 97.64, 97.67, ...
	   97.83, 97.95, 98.05, 97.70, 97.70];

subplot(211);
X = 1:length(Ym1);
plot(X, (Ym1+Ym2)/2, 'r'); hold on
plot(X, (Yq1+Yq2)/2, 'b');
legend('monothread', 'quadrithread');
title('Impact of Multithreading on Learning Speed (Average on 2 Neural Networks)');
xlabel('Training Epochs');
ylabel('Test Result (%)');
xlim([1 length(Ym1)]);
grid on

subplot(212);
X3 = 1:length(Ym3);
plot(X3, Ym3, 'r'); hold on
plot(X3, Yq3, 'b');
legend('monothread', 'quadrithread');
title('Impact of Multithreading on Learning Accuracy');
xlabel('Training Epochs');
ylabel('Test Result (%)');
xlim([1 length(Ym1)]);
grid on


