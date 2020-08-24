import sys, os
import keras
import cv2
import traceback
from args import get_args
from src.keras_utils 			import load_model
from glob 						import glob
from os.path 					import splitext, basename
from src.utils 					import im2single
from src.keras_utils 			import load_model, detect_lp
from src.label 					import Shape, writeShapes
from os import listdir
from os.path import isfile, join


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


args = get_args()

if __name__ == '__main__':

	try:
		
		input_dir  = args.input_dir
		output_dir = args.output_dir
		lp_model = args.lp_model
		print(input_dir)
		lp_threshold = .5

		wpod_net_path = lp_model
		wpod_net = load_model(wpod_net_path)

		#imgs_paths = glob('{}/*.jpg'.format(input_dir))
		onlyfiles = ["{}/{}".format(input_dir,f) for f in listdir(input_dir) if isfile(join(input_dir, f))]
		print(onlyfiles)
		print('Searching for license plates using WPOD-NET')

		for i,img_path in enumerate(onlyfiles):

			print('\t Processing %s' % img_path)

			bname = splitext(basename(img_path))[0]
			Ivehicle = cv2.imread(img_path)

			ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
			side  = int(ratio*288.)
			bound_dim = min(side + (side%(2**4)),608)
			print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

			Llp,LlpImgs,_ = detect_lp(wpod_net,im2single(Ivehicle),bound_dim,2**4,(240,80),lp_threshold)

			if len(LlpImgs):
				Ilp = LlpImgs[0]
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
				Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

				s = Shape(Llp[0].pts)

				cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)
				writeShapes('%s/%s_lp.txt' % (output_dir,bname),[s])

	except:
		traceback.print_exc()
		sys.exit(1)

	sys.exit(0)