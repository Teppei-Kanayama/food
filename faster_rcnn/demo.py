import argparse
import matplotlib.pyplot as plot

import chainer

from chainercv.datasets import voc_bbox_label_names
#from chainercv.links import FasterRCNNVGG16
import sys
sys.path.append("/home/mil/kanayama/workspace/chainercv/chainercv/links/model/faster_rcnn")
from faster_rcnn_vgg import FasterRCNNVGG16

from chainercv import utils
#from chainercv.visualizations import vis_bbox
sys.path.append("/home/mil/kanayama/workspace/chainercv/chainercv/visualizations")
from vis_bbox import vis_bbox
import cupy

from PIL import Image
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1)
    parser.add_argument('--pretrained_model', default='voc07')
    parser.add_argument('--root_path', default="/data/unagi0/food/train_set/")
    parser.add_argument('--image_list', default="/data/unagi0/food/tmp/train1000.txt")
    parser.add_argument('--dst_path', default="/data/unagi0/food/tmp/dst/")
    parser.add_argument('image')
    args = parser.parse_args()

    model = FasterRCNNVGG16(
        n_fg_class=len(voc_bbox_label_names),
        pretrained_model=args.pretrained_model)

    if args.gpu >= 0:
        chainer.cuda.get_device_from_id(args.gpu).use()
        model.to_gpu()

    image_list = open(args.image_list, "r")
    for filename in image_list:
        filename = filename.split('\n')[0]
        print(filename)
        #img = utils.read_image(args.root_path + filename, color=True)
        #img = utils.read_image(args.image, color=True)
        #img = Image.open(args.image)
        img = Image.open(args.root_path + filename)
        img = img.resize((256, 256)) # なぜか、ある特定の形のときエラーになる
        img = np.asarray(img, dtype=np.float32)
        img = np.transpose(img, (2, 0, 1))
        bboxes, labels, scores = model.predict([img])
        bbox, label, score = bboxes[0], labels[0], scores[0]

        vis_bbox(
            img, bbox, label, score, label_names=voc_bbox_label_names)
        #plot.show()
        #plot.savefig(args.dst_path + args.image.split("/")[-1])
        #break
        plot.savefig(args.dst_path + filename)

    image_list.close()

if __name__ == '__main__':
    main()
