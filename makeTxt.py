# coding: utf-8
import os

# 创建Map，这个Map的Key是图片的路径，而他的Value就是Label
dataMap = {}
base_path = os.getcwd() # 获取当前路径
# phonepicture目录下有若干个，表示类别
categories = os.listdir(r'D:\python\project\DSP\CrackFormer-II-main\datasets\CrackLS315\train\valid') # 加入 r 可以不用专门的去处理引号之中的特殊字符

print(categories)
rawdata_path = os.path.join(base_path, r"D:\python\project\DSP\CrackFormer-II-main\datasets\CrackLS315\train\valid")
print(rawdata_path)

for c in categories:
    # a_image_folder 里面存的是当前文件夹的路径，如果再加上图片名就是该图片的绝对路径了，而这正是我们想要的。
    a_image_folder = os.path.join(rawdata_path, c)
    # 某一类别的图片名字(不是绝对路径)都在 image_files 里面了，而c就是他的类别名
    image_files = os.listdir(a_image_folder)
    for image in image_files:
        dataMap[a_image_folder + "\\" + image] = c

# 将其存储为Txt文件
with open(r"D:\python\project\DSP\CrackFormer-II-main\datasets\CrackLS315\train\train.txt", 'w') as f:
    for k, v in dataMap.items():
        # f.write(k + ' ' + v + '\n')
        f.write(k + ' ' + '\n')