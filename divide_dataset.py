import glob
import shutil
from tqdm import tqdm
img_files = glob.glob('VOC2012/JPEGImages/*')
#print(len(img_files))

Val_split = int(0.01 * len(img_files))
print("Validation split = ",Val_split)


for item in tqdm(img_files[0:Val_split]):
  shutil.move(item,"VALIDATION")

import matplotlib.pyplot as plt
import cv2
im = cv2.imread('VOC2012/JPEGImages/2008_001823.jpg')
plt.imshow(im)

import glob
val_files = glob.glob('VALIDATION/*')
print(len(val_files))

import matplotlib.pyplot as plt
import cv2

total_img = 20
num_pr = 5 # number per row

counter = 1
plt.figure(figsize=(35,35))
plt.axis('off')
for item in val_files[:total_img]:
  image = cv2.imread(item, cv2.IMREAD_UNCHANGED)
  plt.subplot(num_pr, num_pr, counter)
  plt.title("Name = {}".format(str(item.split('.')[0])),fontsize=7).set_color('black')
  plt.axis('off')
  plt.imshow(image[:,:,::-1])
  counter += 1
  #break
#plt.show()

plt.savefig('PASCAL_VOC.png')
