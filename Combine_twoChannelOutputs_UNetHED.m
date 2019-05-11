clear all; clc;

str1 = 'unet-tensorflow-keras-region_V3/Trial3/Results_TRY12/';

str2 = 'HED/hed_test/model-49900-side5/';
files1 = dir('unet-tensorflow-keras-region_V3/Trial3/Results_TRY12/*.png');
files2 = dir('HED/hed_test/model-49900-side5/*.png');

file1 = files1';
file2 = files2';


for ifile1 = file1(1:length(file1))
   name1 = strcat(str1, ifile1.name);
   idx = str2num(strtok(ifile1.name, '_*.png'));
   I1 = imread(name1);
   I1 = imfill(I1,'holes');

   [row col dim] = size(I1);
   I1(I1 == 1) = 255;
%    I_rgb = cat(3, I1, I1, I1);
%    I_rgb(:,:,2) = 0;
%    I_rgb(:,:,3) = 0;
%    I1 = I_rgb;

 for k = 1:152
 if str2num(strtok(file2(k).name, '-*.png')) == idx
%      disp "Hello"
%      disp file2(k).name
   ifile2 = file2(k);
   name2 = strcat(str2, ifile2.name);
   I2 = imread(name2);
   [row1 col1 dim1] = size(I2);
   I2_gray = rgb2gray(I2);
   I2_gray = imcomplement(I2_gray);
   
   %I2(I2 ==1) = 255;
   I3(:,:,1) = I1;
   I3(:,:,2) = I2_gray;
   I3(:,:,3) = 0;
  % figure, imshow(I3);
   
   %I2 = rgb2gray(I2);
   %I2 = imbinarize(I2);
   
%    I3 = imoverlay(I1, I2, 'Green');
%    
imwrite(I3, strcat('Combined_channelData_UnetHED_test/', strtok(ifile1.name,'_*'),'.png'));

break;

 else
     continue
 end
 end

end
  
