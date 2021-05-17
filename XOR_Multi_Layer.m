clc
clear
x1=[0,0,1,1];
x2=[0,1,0,1];
y=[0,1,1,0];
epochs = 10000;
lr = 0.1;

input_layer_neuron = 2; 
hidden_layer_neuron = 2; 
output_layer_neuron = 1;

weights_h_x1 = rand(1,hidden_layer_neuron)

weights_h_x2 = rand(1,hidden_layer_neuron);
weights_o_h1 = rand(1,output_layer_neuron);
weights_o_h2 = rand(1,output_layer_neuron);
hidden_bias = rand(1,hidden_layer_neuron)
output_bias = rand(1,output_layer_neuron);
error_list = [];

weight_x1_1_list = [];
weight_x1_2_list = [];
weight_x2_1_list = [];
weight_x2_2_list = [];
weight_o_1_list = [];
weight_o_2_list = [];
bias_1_list = [];
bias_2_list = [];
bias_o_list = [];

%Derivative of Activation Function
function z2 = derv_sigmoid(y2)
  z2 = y2*(1-y2);
endfunction

%Activation Function
function z = sigmoid(y)
  z = 1/(1+exp(-y));
endfunction

%Training
for i=1:epochs
  error_avg = 0;
  if mod(i,50) == 0
  weight_x1_1_list = [weight_x1_1_list;weights_h_x1(1)];
  weight_x1_2_list = [weight_x1_2_list;weights_h_x1(2)];
  weight_x2_1_list = [weight_x2_1_list;weights_h_x2(1)];
  weight_x2_2_list = [weight_x2_2_list;weights_h_x2(2)];
  weight_o_1_list = [weight_o_1_list;weights_o_h1(1)];
  weight_o_2_list = [weight_o_2_list;weights_o_h2(1)];
  bias_1_list = [bias_1_list;hidden_bias(1)];
  bias_2_list = [bias_2_list;hidden_bias(2)];
  bias_o_list = [bias_o_list;output_bias(1)];
  endif
  for j=1:4
    hidden_activation_1 = weights_h_x1(1)*x1(j)+weights_h_x2(1)*x2(j)+hidden_bias(1);
    hidden_output_1 = sigmoid(hidden_activation_1);
    
    hidden_activation_2 = weights_h_x1(2)*x1(j)+weights_h_x2(2)*x2(j)+hidden_bias(2);
    hidden_output_2 = sigmoid(hidden_activation_2);
    
    output_activation = weights_o_h1(1)*hidden_output_1+weights_o_h2(1)*hidden_output_2+output_bias(1);
    predicted_output = sigmoid(output_activation);
    
    error = (y(j)-predicted_output);
    d_error_output = error * derv_sigmoid(predicted_output);
    
    d_error_hidden_x1 = d_error_output*weights_o_h1*derv_sigmoid(hidden_output_1);
    d_error_hidden_x2 = d_error_output*weights_o_h2*derv_sigmoid(hidden_output_2);
    error_avg += abs(error);
    
    %updating the weights
    weights_o_h1 += d_error_output*hidden_output_1*lr;
    weights_o_h2 += d_error_output*hidden_output_2*lr;
    output_bias += d_error_output*lr;
    weights_h_x1(1) += d_error_hidden_x1*lr*x1(j);
    weights_h_x1(2) += d_error_hidden_x2*lr*x1(j);
    weights_h_x2(1) += d_error_hidden_x1*lr*x2(j);
    weights_h_x2(2) += d_error_hidden_x2*lr*x2(j);
    hidden_bias(1) += d_error_hidden_x1*lr;
    hidden_bias(2) += d_error_hidden_x2*lr;
  endfor
  error_list = [error_list;abs(error_avg*0.25)];
  endfor

%Evaluation
for j=1:4
    hidden_activation_1 = weights_h_x1(1)*x1(j)+weights_h_x2(1)*x2(j)+hidden_bias(1);
    hidden_output_1 = sigmoid(hidden_activation_1);
    
    hidden_activation_2 = weights_h_x1(2)*x1(j)+weights_h_x2(2)*x2(j)+hidden_bias(2);
    hidden_output_2 = sigmoid(hidden_activation_2);
    
    output_activation = weights_o_h1(1)*hidden_output_1+weights_o_h2(1)*hidden_output_2+output_bias(1);
    predicted_output = sigmoid(output_activation);
    predicted_output
endfor 
 
%Loss vs Epochs
figure(1)
set(gca, "fontsize", 25)
x = linspace(1,length(error_list),length(error_list));
plot(x,error_list','r')
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
set(gca, "fontsize", 25)
for i=1:length(bias_1_list)
p1 = plot(x, (-weight_x1_1_list(i)*x-bias_1_list(i))/weight_x2_1_list(i),'r');
hold on
p2 = plot(x, (-weight_x1_2_list(i)*x-bias_2_list(i))/weight_x2_2_list(i),'r');
hold on
p3 = plot(x, (-weight_o_1_list(i)*x-bias_o_list(i))/weight_o_2_list(i),'r');
pause(0.01);
if i < length(bias_1_list)
  delete(p1);
  delete(p2);
  delete(p3);
  endif
endfor
