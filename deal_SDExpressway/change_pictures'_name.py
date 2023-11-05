import os.path

def rename(img_folder0,img_folder1,num):
    for img_name in os.listdir(img_folder0):  # os.listdir()： 列出路径下所有的文件
        #os.path.join() 拼接文件路径
        src = os.path.join(img_folder0, img_name)   #src：要修改的目录名
        dst = os.path.join(img_folder1,  str(num) + '.jpg') #dst： 修改后的目录名      注意此处str(num)将num转化为字符串,继而拼接
        os.rename(src, dst) #用dst替代src
        print("Rename success: ", format(img_folder1+'/'+str(num) + '.jpg'))
        num= num+1


def main():
    img_folder0 = 'images_test2' #图片的文件夹路径    直接放你的文件夹路径即可
    img_folder1 = 'images_test3'
    num=4430
    rename(img_folder0,img_folder1,num)

if __name__=="__main__":
    main()

# 得用调试才能跑通？？？