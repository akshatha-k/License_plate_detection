import os
import sys
import traceback
from os import listdir
from os.path import isfile, join
from os.path import splitext, basename
from statistics import mean

import cv2

from args import get_args
from src.label import Shape, writeShapes
from src.tflite_utils import load_model, detect_lp
from src.utils import im2single, get_model_memory_usage, get_logger, setup_dirs, get_gzipped_model_size


def adjust_pts(pts, lroi):
    return pts * lroi.wh().reshape((2, 1)) + lroi.tl().reshape((2, 1))


setup_dirs()
logger = get_logger("lp-tflite-detection")
args = get_args()

if __name__ == '__main__':
    args = get_args()
    if args.use_colab:
        from google.colab import drive

        drive.mount('/content/gdrive')
        OUTPUT_DIR = '/content/gdrive/My Drive/lpd/{}_{}_{}_{}_{}'.format(args.image_size, args.epochs, args.prune_model,
                                                                       args.initial_sparsity,
                                                                       args.final_sparsity)
        if not os.path.isdir(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
        tflite_path = '{}/{}.tflite'.format(OUTPUT_DIR, args.model)
        pruned_tflite_path = '{}/{}_pruned.tflite'.format(OUTPUT_DIR, args.model)
        test_dir = '/content/gdrive/My Drive/lpd/test_images'
        output_dir = '{}/{}'.format(OUTPUT_DIR, 'tflite_results')
        if not os.path.isdir(output_dir): os.makedirs(output_dir)
        pruned_output_dir = '{}/{}_pruned'.format(OUTPUT_DIR, 'tflite_results')
        if not os.path.isdir(pruned_output_dir): os.makedirs(pruned_output_dir)
try:
    lp_threshold = .5
    inference_times = []
    wpod_net_path = tflite_path
    wpod_net = load_model(wpod_net_path)
    onlyfiles = ["{}/{}".format(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f))]
    print(onlyfiles)
    print('Searching for license plates using WPOD-NET')
    logger.info('Searching for license plates using WPOD-NET')

    for i, img_path in enumerate(onlyfiles):

        print('\t Processing %s' % img_path)
        logger.info('\t Processing %s' % img_path)

        bname = splitext(basename(img_path))[0]
        Ivehicle = cv2.imread(img_path)

        ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
        side = int(ratio * 288.)
        bound_dim = min(side + (side % (2 ** 4)), 608)
        #print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
        #logger.info("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

        Llp, LlpImgs, elapsed_time = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2 ** 4, (240, 80),
                                               lp_threshold)
        inference_times.append(elapsed_time)

        if len(LlpImgs):
            Ilp = LlpImgs[0]
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
            Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

            s = Shape(Llp[0].pts)

            cv2.imwrite('%s/%s_lp.png' % (output_dir, bname), Ilp * 255.)
            writeShapes('%s/%s_lp.txt' % (output_dir, bname), [s])
    print("Model size after gzip is : {} bytes".format(get_gzipped_model_size(tflite_path)))
    logger.info("Model size after gzip is : {} bytes".format(get_gzipped_model_size(tflite_path)))
    print("Mean inference time (in seconds) : {}".format(mean(inference_times)))
    logger.info("Mean inference time (in seconds) : {}".format(mean(inference_times)))

    if args.prune_model:
        lp_threshold = .5
        inference_times = []
        wpod_net_path = pruned_tflite_path
        wpod_net = load_model(wpod_net_path)
        onlyfiles = ["{}/{}".format(test_dir, f) for f in listdir(test_dir) if isfile(join(test_dir, f))]
        print(onlyfiles)
        print('Searching for license plates using WPOD-NET(PRUNED) ')
        logger.info('Searching for license plates using WPOD-NET(PRUNED) ')

        for i, img_path in enumerate(onlyfiles):

            print('\t Processing %s' % img_path)
            logger.info('\t Processing %s' % img_path)

            bname = splitext(basename(img_path))[0]
            Ivehicle = cv2.imread(img_path)

            ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
            side = int(ratio * 288.)
            bound_dim = min(side + (side % (2 ** 4)), 608)
            # print("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))
            # logger.info("\t\tBound dim: %d, ratio: %f" % (bound_dim, ratio))

            Llp, LlpImgs, elapsed_time = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, 2 ** 4, (240, 80),
                                                   lp_threshold)
            inference_times.append(elapsed_time)

            if len(LlpImgs):
                Ilp = LlpImgs[0]
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
                Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

                s = Shape(Llp[0].pts)

                cv2.imwrite('%s/%s_lp.png' % (pruned_output_dir, bname), Ilp * 255.)
                writeShapes('%s/%s_lp.txt' % (pruned_output_dir, bname), [s])
        print("(PRUNED) Model size after gzip is : {} bytes".format(get_gzipped_model_size(pruned_tflite_path)))
        logger.info("(PRUNED) Model size after gzip is : {} bytes".format(get_gzipped_model_size(pruned_tflite_path)))
        print("(PRUNED) Mean inference time (in seconds) : {}".format(mean(inference_times)))
        logger.info("(PRUNED) Mean inference time (in seconds) : {}".format(mean(inference_times)))
except:
    traceback.print_exc()
    sys.exit(1)

sys.exit(0)
