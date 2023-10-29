clc; clear; close all; warning off all;
%%proses pelatihan
%menetapkan lokasi folder
nama_folder = 'data latih';
%%membaca file berekstensi jpg
nama_file = dir(fullfile(nama_folder,'*.jpg'));
%membaca jumlah file yang berekstensi .jpg
jumlah_file = numel(nama_file);

%menginisialisasi variable ciri latih
ciri_latih = zeros(jumlah_file,4);
%melakukan pengolahan citra terhadap file
for n = 1:jumlah_file
    %membacafile citra rgb
    Img = imread(fullfile(nama_folder,nama_file(n).name));
%     figure, imshow(Img)
    %melakukan konversi citra rgb ke grayscale
    Img_gray = rgb2gray(Img);
%     figure, imshow(Img_gray)
    %melakukan konversi citra grayscale ke biner
    bw = imbinarize(Img_gray);
%     figure, imshow(bw)
    %melkukan operasi komplemen
    bw = imcomplement(bw);
%     figure, imshow(bw)
    %melakukan operasi morfologi filing holes
    bw = imfill(bw,'holes');
%     figure, imshow(bw)
    %ekstrasi ciri
    %melakukan konversi citra rgb ke hsv
    HSV = rgb2hsv(Img);
%     figure, imshow(HSV)
    %melakukan ekstrasi komponen h,s &v pada citra hsv
    H = HSV(:,:,1);
    S = HSV(:,:,2);
    V = HSV(:,:,3);
    %mengubah nilai pixel bg menjadi 0
    H(~bw)=0;
    S(~bw)=0;
    V(~bw)=0;
    %menghitung nilai rata" h,s,v
    Hue = sum(sum(H))/sum(sum(bw));
    Saturation = sum(sum(S))/sum(sum(bw));
    Value = sum(sum(V))/sum(sum(bw));
    %menghitung luas objek
    Luas = sum(sum(bw));
    %mengisi variable ciri latih dengan ciri hasil ekstrasi
    ciri_latih(n,1)= Hue;
    ciri_latih(n,2)= Saturation;
    ciri_latih(n,3)= Value;
    ciri_latih(n,4)= Luas;
end

%menyusun kelas latih
kelas_latih = cell(jumlah_file,1);
%mengisi nama" kendaraan
for k = 1:3
    kelas_latih{k}= 'mobil';
end
for k = 4:6
    kelas_latih{k}= 'motor';
end
for k = 7:9
    kelas_latih{k}= 'sepeda';
end


%klasifikasi citra menggunakan algoritma naive bayes
Mdl = fitcnb(ciri_latih,kelas_latih);

%membaca kelas keluaran hasil pelatihan
hasil_latih = predict(Mdl,ciri_latih);

% % menghitung akurasi pelatihan
jumlah_benar = 0;
for k = 1:jumlah_file
     if isequal(hasil_latih{k},kelas_latih{k})
         jumlah_benar = jumlah_benar+1;
     end
 end
% 
 akurasi_pelatihan = jumlah_benar/jumlah_file*100
