import os
import sys
import traceback
from os import listdir
from os.path import isfile, join
from os.path import splitext, basename

import cv2

from args import get_args
from src.keras_utils import load_model, detect_lp
from src.label import Shape, writeShapes
from src.utils import im2single


def adjust_pts(pts, lroi):
    return pts * lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


args = get_args()

if __name__ == '__main__':
    args = get_args()
    if args.use_colab:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}'.format(args.image_size, args.initial_sparsity,
                                                                    args.final_sparsity)
        if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        lp_model = '%s/%s_trained' % (OUTPUT_DIR, args.model)
        test_dir = '/content/gdrive/My Drive/lpd/test_images'
        output_dir = '{}/results'.format(OUTPUT_DIR)
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
try:
    lp_threshold = .5

    wpod_net_path = lp_model
    wpod_net = load_model(wpod_net_path)
    onlyfiles = ["{}/{}".format(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f))]
    print(onlyfiles)
    print('Searching for license plates using WPOD-NET')

    for i, img_path in enumerate(onlyfiles):

        print('\t Processing %s' % img_path)

        bname = splitext(basename(img_path))[0]
        Ivehicle = cv2.imread(img_path)

        ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 4)), 608)
        print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

        Llp, LlpImgs, _ = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2 ** 4, (240, 80), lp_threshold)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)
            print("here")
            cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp * 255.)
            writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])

except:
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)
