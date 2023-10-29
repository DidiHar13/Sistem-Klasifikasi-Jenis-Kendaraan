clc; clear; close all; 

%%proses pelatihan
%menetapkan lokasi folder
nama_folder = 'data latih';
%%membaca file berekstensi jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
%membaca jumlah file yang berekstensi .jpg
jumlah_file = numel(nama_file);
%menginisialisasi variable ciri latih
ciri = zeros(jumlah_file,5);
kelas = zeros(jumlah_file,1);
%melakukan ekstrasi ciri terhadap file
for n = 1:jumlah_file
    %membaca file citra
    Img = imread(fullfile(nama_folder,nama_file(n).name));
    %mengkonversi citra rgb ke hsv
    hsv = rgb2hsv(Img);
    %mengekstrasi komponen v
    V = hsv(:,:,3);
    %melakukan tresholding terhadap komponen value
    bw = IMBINARIZE(,0.9);
    %melakukan median filtering
    bw = medfilt2(~bw,[5,5]);
    %melakukan operasi morfologi filing holes
    bw = imfill(bw,'holes');
    %melakukan morfologi area opening
    bw = bwareaopen(bw,1000);
    %melakukan morfologi closing
    str = strel('square', 10);
    bw = imdilate(bw,str);
    %melakukan ekstrasi ciri terhadap citra biner hasil tresholding
    s = regionprops(bw, 'all');
    area = cat(1,s.Area);
    perimeter = cat(1,s.Perimeter);
    eccentricity = cat(1,s.Eccentricity);
    mayor = cat(1,s.MajorAxisLength);
    minor = cat(1,s.MinorAxisLength);
    %menyusun variable ciri
    ciri(n,1) = area;
    ciri(n,2) = perimeter;
    ciri(n,3) = eccentriicity;
    ciri(n,4) = mayor;
    ciri(n,5) = minor;
end
% menetapkan kelas target latih
kelas(1:10) = 1;
kelas(11:20) = 2;
kelas(21:30) = 3;
 
% menyusun data latih
data_training = [kelas,ciri];
% menyusun parameter2 elm
NumberofInputNeurons = 5;
NumberofHiddenNeurons = 60;
% bobot diinisialisasi secara random
% InputWeight = rand(NumberofHiddenNeurons,NumberofInputNeurons)*2-1;
% BiasofHiddenNeurons = rand(NumberofHiddenNeurons,1);
% bobot ditetapkan di awal
load bobot_awal
Elm_Type = 1;
ActivationFunction  = 'sin';
 
% pelatihan elm
[~, ~, ~, ~, predicted_class] = ...
    ELM(data_training, data_training, ...
    InputWeight, BiasofHiddenNeurons, Elm_Type,...
    ActivationFunction);
 
% menghitung akurasi pelatihan
[~,n] = find(predicted_class==kelas);
akurasi = numel(n)/jumlah_file*100;
disp(['akurasi pelatihan = ',num2str(akurasi),'%'])
 
% menyimpan variabel2 pelatihan
save('net','data_training','InputWeight','BiasofHiddenNeurons',...
    'Elm_Type','ActivationFunction')