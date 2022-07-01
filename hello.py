import os
print(os.getcwd())#当前文件所在的路径
print(os.listdir(os.getcwd())) #在/kaggle/working文件夹下有两个隐藏文件
print(os. listdir("../../" ))
print(os. listdir("../"))
print(os.listdir("../input")) #里面是没有数据的
print(os.listdir("../../kaggle"))