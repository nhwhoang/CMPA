close all; clear all;
set(0,'DefaultFigureWindowStyle','docked')

%% Generate data
Is = 0.01e-12;
Ib = 0.1e-12;
Vb = 1.3;
Gp = 0.1;

V = linspace(-1.95,0.7,200);

I = Is*(exp((1.2/0.025).*V)-1) + Gp.*V - Ib*(exp((-1.2/0.025).*(V+Vb))-1);

% +/- 20% noise on I 
noise = 0.4*rand(1,200) - 0.2;

Inoise = I.*(1+noise);

figure(1)
subplot(2,1,1)
plot(V,I);  hold on;
plot(V,Inoise)
legend('I','Inoise');
xlabel('voltage (V)')
ylabel('current (A)')

subplot(2,1,2)
semilogy(V,abs(I));  hold on;
semilogy(V,abs(Inoise))
legend('I','Inoise');
xlabel('voltage (V)')
ylabel('log current (A)')



%% Polynomial fitting
fit4 = polyfit(V,I,4);    I4 = polyval(fit4,V);
fit8 = polyfit(V,I,8);    I8 = polyval(fit8,V);

fitnoise4 = polyfit(V,Inoise,4);      Inoise4 = polyval(fitnoise4,V);
fitnoise8 = polyfit(V,Inoise,8);      Inoise8 = polyval(fitnoise8,V);


figure(2)
subplot(2,1,1)
plot(V,I); hold on;
plot(V,I4); hold on;
plot(V,I8); hold on;
plot(V,Inoise); hold on;
plot(V,Inoise4); hold on;
plot(V,Inoise8);
legend('I','I4','I8','Inoise','Inoise4','Inoise8');
title('Polynomial Fit');
xlabel('voltage (V)');
ylabel('current (A)');

subplot(2,1,2)
semilogy(V,abs(I)); hold on;
semilogy(V,abs(I4)); hold on;
semilogy(V,abs(I8)); hold on;
semilogy(V,abs(Inoise)); hold on;
semilogy(V,abs(Inoise4)); hold on;
semilogy(V,abs(Inoise8));
legend('I','I4','I8','Inoise','Inoise4','Inoise8');
title('Polynomial Fit');
xlabel('voltage (V)');
ylabel('log current (A)');



%% Nonlinear curve fitting
% fitting A and C
foa = fittype('A.*(exp(1.2*x/25e-3)-1) + Gp.*x - C*(exp(1.2*(-(x+Vb))/25e-3)-1)');
ffa = fit(V.',I.',foa);
Ifa = ffa(V).';

% fitting A, B and C
fob = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+Vb))/25e-3)-1)');
ffb = fit(V.',I.',fob);
Ifb = ffb(V).';

% fitting all A, B, C and D
foc = fittype('A.*(exp(1.2*x/25e-3)-1) + B.*x - C*(exp(1.2*(-(x+D))/25e-3)-1)');
ffc = fit(V.',I.',foc);
Ifc = ffc(V).';


figure(3)
subplot(2,1,1);
plot(V,I); hold on;
plot(V,Ifa); hold on;
plot(V,Ifb); hold on;
plot(V,Ifc);
title('Nonlinear Curve fit')
legend('I','Ifa','Ifb','Ifc')
xlabel('voltage (V)');
ylabel('current (A)');

subplot(2,1,2);
semilogy(V,abs(I)); hold on;
semilogy(V,abs(Ifa)); hold on;
semilogy(V,abs(Ifb)); hold on;
semilogy(V,abs(Ifc));
title('Nonlinear Curve fit')
legend('I','Ifa','Ifb','Ifc')
xlabel('voltage (V)');
ylabel('log current (A)');



%% Neural Net model fitting
inputs = V.';
targets = I.';
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
outputs = net(inputs);
errors = gsubtract(outputs,targets);
performance = perform(net,targets,outputs);
view(net);
Inn = outputs;


figure(4)
subplot(2,1,1)
plot(V,I); hold on;
plot(V,Inn)
title('Neural Net model fit')
legend('I','Inn')
xlabel('voltage (V)');
ylabel('current (A)');

subplot(2,1,2)
semilogy(V,abs(I)); hold on;
semilogy(V,abs(Inn))
title('Neural Net model fit')
legend('I','Inn')
xlabel('voltage (V)');
ylabel('log current (A)');

