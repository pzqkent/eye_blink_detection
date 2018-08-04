input = load('ear_fortrain.txt')
input1 = load('ear_after_threshold.txt')
input2 = load('ear_after_svm.txt')

subplot(3,1,1)
plot(input,'LineWidth',2)
xlim([356,406])
xlabel('frame')
ylabel('EAR')

subplot(3,1,2)
plot(input1,'LineWidth',2)
xlim([350,400])
xlabel('frame')
ylabel('Output after threshold')
title('Threshold = 0.22')

subplot(3,1,3)
plot(input2,'LineWidth',2)
xlim([350,400])
xlabel('frame')
ylabel('Predict from SVM')

% xlim([350,400])

% ylim([0,0.35])