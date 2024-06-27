clc
clear all

% Tablo oluştururken Türkçe karakterlerin hata vermemesi için 'preserve'
% kullanıldı
opts = detectImportOptions('veri.xlsx');
opts.VariableNamingRule = 'preserve';

% Veriyi oku
Data = readtable('veri.xlsx');
disp(Data);

% Giriş ve çıkış değişkenlerini ayır
X = table2array(Data(:, 1:22)); % Giriş değişkenleri
y = table2array(Data(:, 23));   % Çıkış değişkeni

% Veri görselleştirme
figure;

% Kemik Erimesi Dağılımı (Histogram)
subplot(3, 2, 1);
histogram(y);
title('Kemik Erimesi Histogram');
xlabel('Kemik Erimesi');
ylabel('Frekans');

% Giriş Değişkenlerinin Dağılımı (Histogramlar)
subplot(3, 2, 2);
histogram(X(:, 1));
title('Yaş');
xlabel(['Değerler - ', Data.Properties.VariableNames{1}]);
ylabel('Frekans');

subplot(3, 2, 3);
histogram(X(:, 2));
title('Meslek');
xlabel(['Değerler - ', Data.Properties.VariableNames{2}]);
ylabel('Frekans');

subplot(3, 2, 4);
histogram(X(:, 3));
title('Kilo');
xlabel(['Değerler - ', Data.Properties.VariableNames{3}]);
ylabel('Frekans');

subplot(3, 2, 5);
histogram(X(:, 4));
title('Boy');
xlabel(['Değerler - ', Data.Properties.VariableNames{4}]);
ylabel('Frekans');

subplot(3, 2, 6);
histogram(X(:, 5));
title('Cinsiyet');
xlabel(['Değerler - ', Data.Properties.VariableNames{5}]);
ylabel('Frekans');

% Kutu Grafiği
figure;
boxplot(X);
title('Giriş Değişkenleri Kutu Grafiği');
xlabel('Değişkenler');
ylabel('Değerler');

% Korelasyon Matrisi oluşturma
corrMatrix = corr(X,y);
% Korelasyon Matrisi oluşturma

corrMatrix2 = corr(X);

% Korelasyon Matrisi
figure;
heatmap(corrMatrix, 'ColorbarVisible', 'on');
title('Korelasyon Matrisi');

figure;
heatmap(corrMatrix2, 'ColorbarVisible', 'on');
title('Korelasyon Matrisi');

% Yaş sütunu ve kemik erimesi sütununu al
ages = Data.yas;
boneDensity = Data.kemik_erimesi;

% Scatter plot çizimi
figure;
scatter(boneDensity, ages, 'o', 'filled', 'MarkerFaceColor', 'b');
title('Yaşa Göre Kemik Erimesi');
xlabel('Kemik Erimesi');
ylabel('Yaş');

%trigliserit
trigliserit = Data.trigliserit;
boneDensity = Data.kemik_erimesi;

% Scatter plot çizimi
figure;
scatter(boneDensity, trigliserit, 'o', 'filled', 'MarkerFaceColor', 'b');
title('Trigliserite Göre Kemik Erimesi');
xlabel('Kemik Erimesi');
ylabel('Trigliserit');

% Normalizasyon (Min-Max Normalizasyonu)
minVals = min(X);
maxVals = max(X);
X_normalized = (X - minVals) ./ (maxVals - minVals);


% Veriyi eğitim ve test olarak ayır
rng(1); % Tekrarlanabilirlik için rastgele sayı üretme tohumu ayarla
cv = cvpartition(size(X,1), 'HoldOut', 0.3); % Veriyi 70% eğitim, 30% test olarak ayır
X_train = X(cv.training,:);
y_train = y(cv.training,:);
X_test = X(cv.test,:);
y_test = y(cv.test,:);

% Yapay Sinir Ağı Modeli Oluşturma
hiddenLayerSizes = [15 20 15]; % Gizli katman boyutları
net = feedforwardnet(hiddenLayerSizes, 'trainlm');

% Eğitim parametrelerini ayarla
net.trainParam.show = 20;
net.trainParam.epochs = 20;
net.trainParam.goal = 1e-25;

% Modeli eğit
[net,tr] = train(net, X_train', y_train');

% Test verisi üzerinde tahmin yap
predictedOutput = sim(net, X_test');

% Hata hesapla
error = gsubtract(y_test', predictedOutput);
mseError = mean(error.^2);
disp(['Test Seti İçin Mean Squared Error: ', num2str(mseError)]);

% Confusion matrixi çiz
figure;
plotconfusion(y_test', predictedOutput);
title("Test Data");

% Eğitim verisi üzerinde tahmin yap
predictedTrainOutput = sim(net, X_train');

% Eğitim verisi üzerindeki hata hesapla
trainError = gsubtract(y_train', predictedTrainOutput);
trainMSEError = mean(trainError.^2);
disp(['Eğitim Seti İçin Mean Squared Error: ', num2str(trainMSEError)]);

% Confusion matrixi çiz
figure;
plotconfusion(y_train', predictedTrainOutput);
title("Train Data");