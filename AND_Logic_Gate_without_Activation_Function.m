clc
clear
x1=[0,0,1,1];
x2=[0,1,0,1];
y=[0,0,0,1];
epochs = 100;
lr = 0.1;

input_layer_neuron = 2; 
output_layer_neuron = 1;
error_list = [];
weights = rand(1,input_layer_neuron);
bias = rand(1,1);

weight_1_list = [];
weight_2_list = [];
bias_list = [];

%Training 
for i=1:epochs
  error_avg = 0;
  weight_1_list = [weight_1_list;weights(1)];
  weight_2_list = [weight_2_list;weights(2)];
  bias_list = [bias_list;bias(1)];
  for j=1:4
    predicted_output = weights(1)*x1(j)+weights(2)*x2(j)+bias(1);
    
    error = (y(j)-predicted_output);
    error_avg += abs(error);
    
    %updating the weights
    weights(1) += error*x1(j)*lr;
    weights(2) += error*x2(j)*lr;
    bias(1) += error*lr;
  endfor
  error_list = [error_list;abs(error_avg*0.25)];
endfor

%Evaluation
for j=1:4
    predicted_output = weights(1)*x1(j)+weights(2)*x2(j)+bias(1);
    predicted_output
endfor

%Loss vs Epochs
figure(1)
x = linspace(1,length(error_list),length(error_list));
plot(x,error_list')
title ("Loss vs Epochs");
xlabel ("Epochs");
ylabel ("Loss");
set(gca, "fontsize", 25)

%Visualization
x = linspace(0,1,10);
figure(2)
scatter(x1,x2,'filled')
hold on
axis tight
axis equal
title ("Learning Visualization");
xlabel ("X2");
ylabel ("X1");
set(gca, "fontsize", 25)
for i=1:length(bias_list)
p = plot(x, (-weight_1_list(i)*x-bias_list(i))/weight_2_list(i),'r');
pause(0.01);
if i < length(bias_list)
  delete(p);
  endif
endfor
