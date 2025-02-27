clc;
clear;

accuracy = [];
n = 5;  % 每个人10张照片,5个训练集,5个测试集

function accuracy = Identify(train_data, test_data, label)
    count = 0;
    [M, ~] = size(train_data);  % 读取训练集大小,M=40
    [m, ~] = size(test_data);   % 读取测试集大小

    [coeff, ~, ~] = pca(train_data);    % 使用pca训练数据
    pca_train = train_data * coeff; % 把训练的数据映射到k维空间上

    for i = 1:m
        distance = [];
        test_pca = test_data(i,:) * coeff;
        for k = 1:M   % 遍历每个人像的特征向量
            distance = [distance,norm(pca_train(k,:) - test_pca, 2)];   %  计算与被测图片距离(2范数)，距离越小越相似
        end

        [~, index] = min(distance);   % 从所有可能种类中找出相似度最高的向量
        if index == label(i)
            count = count + 1;
        end
    end
    accuracy = count / m;   % 直接计算整体准确率
end

function [train_data, test_data, label] = divide_data(n)  
    train_data = zeros(40,10304);   % 将每个图像展平即为92 * 112 = 10304维
    test_data = [];
    label = [];
    
    % 遍历每个人
    for i = 1:40
        randnum = randperm(10, n);  % 从每个人的10张照片中选出n=5张作为训练集
        for j = 1:10
            img = reshape(double(imread(['./att_faces/s', num2str(i), '/', num2str(j), '.pgm'] ...
                                        )),1,10304); 
            if ismember(j, randnum)
                train_data(i, :) = train_data(i, :) + img;    % 填充训练集i行对应第i个人 
            else
                test_data = [test_data; img];
                label = [label; i];   % 记录测试集和测试集对应的标签
            end
        end
    end
    train_data = train_data / n;    % 将训练图像平均成一张特征图像
end

rounds = 20;    % 训练轮数，避免训练的随机性
for i = 1:rounds
    [train_data, test_data, label] = divide_data(n);
    acc = Identify(train_data, test_data, label);
    accuracy = [accuracy, acc];
    % 打印当前迭代的准确率
    fprintf('Iteration %d Accuracy: %.4f\n', i, acc);
end

max_acc = max(accuracy);    % 找出最大的正确率
mean_acc = mean(accuracy);  % 找出平均正确率
fprintf('max=%f, min=%f\n',max_acc, mean_acc);

figure;

x = 1:rounds;
plot(x, accuracy,'r--o');
% plot();
ylim([0.8,1]);
xlabel('times');
ylabel('accuracy');
title(['max=',num2str(max_acc), 'mean=', num2str(mean_acc)]);