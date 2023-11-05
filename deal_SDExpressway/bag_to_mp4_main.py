import glob
import sys
import glob
import os
import argparse

def mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='''This is a code for bag2video.''')
    parser.add_argument('--data_root', type=str, default=r'bags', help='path to the root of data')
    parser.add_argument('--out_root', type=str, default=r'bag_to_videos', help='path to the root of data')
    parser.add_argument('--fps', type=int, default=1, help='path to the root of data')
    args = parser.parse_args()

    mkdir(args.out_root)
    bagList = glob.glob(os.path.join(args.data_root, "*.bag"))
    for bagPath in bagList:
        baseName = os.path.basename(bagPath).split(".")[0]
        outPath = os.path.join(args.out_root, baseName + ".mp4")
        os.system("python bag_to_mp4.py -o {outName} --fps {fps} {bagPath}".format(outName=outPath, fps=args.fps, bagPath=bagPath))
