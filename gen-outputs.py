from os import listdir
from os.path import isfile, join

import cv2

from args import get_args
from src.drawing_utils import draw_label

args = get_args()

YELLOW = (0, 255, 255)
RED = (0, 0, 255)

input_dir = args.test_dir
output_dir = args.output_dir

inp_files = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
lp_files = [f for f in listdir(output_dir) if isfile(join(output_dir, f))]

for i, inp_path in enumerate(inp_files):
    I = cv2.imread("{}/{}".format(input_dir, inp_path))

    # draw_label(I,lcar,color=YELLOW,thickness=3)
    lp_label = "{}/{}_lp.txt".format(output_dir, inp_path[:-4])
    print(inp_path)
    print(lp_label)
    draw_label(I, lcar, color=YELLOW, thickness=3)
    # if isfile("{}/{}".format(input_dir,inp_path)):

    # 	Llp_shapes = readShapes(lp_label)
    # 	pts = Llp_shapes[0].pts*lcar.wh().reshape(2,1) + lcar.tl().reshape(2,1)
    # 	ptspx = pts*np.array(I.shape[1::-1],dtype=float).reshape(2,1)
    # 	draw_losangle(I,ptspx,RED,3)
    # 	'''
    # 	if isfile(lp_label_str):
    # 		with open(lp_label_str,'r') as f:
    # 			lp_str = f.read().strip()
    # 		llp = Label(0,tl=pts.min(1),br=pts.max(1))
    # 		write2img(I,llp,lp_str)

    # 		sys.stdout.write(',%s' % lp_str)
    # 		'''
    cv2.imwrite("{}/{}_output.png".format(output_dir, inp_path[:-4]), I)
