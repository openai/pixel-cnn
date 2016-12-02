import cifar10_data
import argparse
import plotting
import numpy as np

data_dir = '/home/tim/data'

parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='./log')
parser.add_argument('--data_dir', type=str, default='/home/tim/data')
parser.add_argument('--plot_title', type=str, default=None)
args = parser.parse_args()
print(args)

data_dir = args.data_dir

trainx, trainy = cifar10_data.load(data_dir)

ids = [[] for i in range(10)]
for i in range(trainx.shape[0]):
    if len(ids[trainy[i]]) < 10:
        ids[trainy[i]].append(i)
    if np.alltrue(np.asarray([len(_ids) >= 10 for _ids in ids])):
        break

images = np.zeros((10*10,32,32,3),dtype='uint8')
for i in range(len(ids)):
    for j in range(len(ids[i])):
        images[10*j+i] = trainx[ids[i][j]].transpose([1,2,0])
print(ids)

img_tile = plotting.img_tile(images, aspect_ratio=1.0, border_color=1.0, stretch=True)
img = plotting.plot_img(img_tile, title=args.plot_title if args.plot_title != 'None' else None)
plotting.plt.savefig(args.save_dir + '/cifar10_orig_images.png')
plotting.plt.close('all')

