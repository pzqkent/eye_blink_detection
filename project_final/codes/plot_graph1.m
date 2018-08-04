input = load('ear_fortrain1.txt')
input1 = load('ear_after_threshold1.txt')
input2 = load('ear_after_svm1.txt')

subplot(3,1,1)
plot(input,'LineWidth',2)
xlim([1000+6,1100+6])
xlabel('frame')
ylabel('EAR')

subplot(3,1,2)
plot(input1,'LineWidth',2)
xlim([1000,1100])
xlabel('frame')
ylabel('Output after threshold')
title('Threshold = 0.22')

subplot(3,1,3)
plot(input2,'LineWidth',2)
xlim([1000,1100])
xlabel('frame')
ylabel('Predict from SVM')

% xlim([350,400])

% ylim([0,0.35])