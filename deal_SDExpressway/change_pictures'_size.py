import os
from PIL import Image

    
 
 
if __name__ == '__main__':
    #  待处理图片路径下的所有文件名字
    all_file_names = os.listdir('./images_test/')
    for file_name in all_file_names:
        #  待处理图片路径
        img_path = Image.open(f'./images_test/{file_name}')
        #  resize图片大小，入口参数为一个tuple，新的图片的大小
        img_size = img_path.resize((1280, 720))
        #  处理图片后存储路径，以及存储格式
        img_size.save(f'./images_test2/{file_name}', 'JPEG')
        print('图片保存地址: ',format("images_test2/"+file_name) )
    print("finish")

# 这个代码得用调试才能跑通，不能直接编译？？？