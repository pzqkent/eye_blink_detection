input = load('ear_fortrain2.txt')
input1 = load('ear_after_threshold2.txt')
input2 = load('ear_after_svm2.txt')

subplot(3,1,1)
plot(input,'LineWidth',2)
xlim([600+6,800+6])
xlabel('frame')
ylabel('EAR')
ylim([0.15,0.4])

subplot(3,1,2)
plot(input1,'LineWidth',2)
xlim([600,800])
xlabel('frame')
ylabel('Output after threshold')
title('Threshold = 0.25')

subplot(3,1,3)
plot(input2,'LineWidth',2)
xlim([600,800])
xlabel('frame')
ylabel('Predict from SVM')

% xlim([350,400])

% ylim([0,0.35])