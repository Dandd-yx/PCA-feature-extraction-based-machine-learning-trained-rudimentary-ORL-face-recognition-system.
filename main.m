clc;
clear;

accuracy = [];
n = 5;  % ÿ����10����Ƭ,5��ѵ����,5�����Լ�

function accuracy = Identify(train_data, test_data, label)
    count = 0;
    [M, ~] = size(train_data);  % ��ȡѵ������С,M=40
    [m, ~] = size(test_data);   % ��ȡ���Լ���С

    [coeff, ~, ~] = pca(train_data);    % ʹ��pcaѵ������
    pca_train = train_data * coeff; % ��ѵ��������ӳ�䵽kά�ռ���

    for i = 1:m
        distance = [];
        test_pca = test_data(i,:) * coeff;
        for k = 1:M   % ����ÿ���������������
            distance = [distance,norm(pca_train(k,:) - test_pca, 2)];   %  �����뱻��ͼƬ����(2����)������ԽСԽ����
        end

        [~, index] = min(distance);   % �����п����������ҳ����ƶ���ߵ�����
        if index == label(i)
            count = count + 1;
        end
    end
    accuracy = count / m;   % ֱ�Ӽ�������׼ȷ��
end

function [train_data, test_data, label] = divide_data(n)  
    train_data = zeros(40,10304);   % ��ÿ��ͼ��չƽ��Ϊ92 * 112 = 10304ά
    test_data = [];
    label = [];
    
    % ����ÿ����
    for i = 1:40
        randnum = randperm(10, n);  % ��ÿ���˵�10����Ƭ��ѡ��n=5����Ϊѵ����
        for j = 1:10
            img = reshape(double(imread(['./att_faces/s', num2str(i), '/', num2str(j), '.pgm'] ...
                                        )),1,10304); 
            if ismember(j, randnum)
                train_data(i, :) = train_data(i, :) + img;    % ���ѵ����i�ж�Ӧ��i���� 
            else
                test_data = [test_data; img];
                label = [label; i];   % ��¼���Լ��Ͳ��Լ���Ӧ�ı�ǩ
            end
        end
    end
    train_data = train_data / n;    % ��ѵ��ͼ��ƽ����һ������ͼ��
end

rounds = 20;    % ѵ������������ѵ���������
for i = 1:rounds
    [train_data, test_data, label] = divide_data(n);
    acc = Identify(train_data, test_data, label);
    accuracy = [accuracy, acc];
    % ��ӡ��ǰ������׼ȷ��
    fprintf('Iteration %d Accuracy: %.4f\n', i, acc);
end

max_acc = max(accuracy);    % �ҳ�������ȷ��
mean_acc = mean(accuracy);  % �ҳ�ƽ����ȷ��
fprintf('max=%f, min=%f\n',max_acc, mean_acc);

figure;

x = 1:rounds;
plot(x, accuracy,'r--o');
% plot();
ylim([0.8,1]);
xlabel('times');
ylabel('accuracy');
title(['max=',num2str(max_acc), 'mean=', num2str(mean_acc)]);