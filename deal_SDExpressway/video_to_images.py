import argparse
import os
import cv2



if __name__ == '__main__':
    # 视频文件和图片保存地址
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_file', required=False, default='bag_to_videos')   # 视频所在的文件夹
    parser.add_argument('--image_file', required=False, default='images_test'+'/')   # 初始图片所在的文件夹
    args = parser.parse_args()
    video_file = args.video_file
    image_file = args.image_file
    list = os.listdir(video_file)   # 获取video文件列表

    name = 0
    for i in range(0, len(list)):
        video_path = os.path.join(video_file, list[i])  # 获取每个mp4文件的绝对路径

        # 设置固定帧率.
        FRAME_RATE = 3

        # 读取视频文件
        videoCapture = cv2.VideoCapture(video_path)
        # 读帧
        success, frame = videoCapture.read()

        j = 0
        while success:
            j = j+1
            # 每隔固定帧保存一张图片
            if j % FRAME_RATE == 0:
                name = name + 1
                pic_address = image_file + str(name) + '.jpg'
                cv2.imwrite(pic_address, frame)
                print('图片保存地址：', image_file + str(name) + '.jpg')
            success, frame = videoCapture.read()

