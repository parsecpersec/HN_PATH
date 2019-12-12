import openslide
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from openslide.deepzoom import DeepZoomGenerator
SlidePath = '/run/media/xuhao/data/HNSCslides'
TilePath = '/run/media/xuhao/data/HNSCtiles'
content = os.walk(SlidePath)
NSample = 0
SampleList, FileList = [], []
for path, dir_list, file_list in content:
    for file in file_list:
        # only '01' for primary tumor
        if file.endswith('.svs') and file[13:15] == '01' and file[0:12] not in SampleList:
            SampleList.append(file[0:12])
            NSample += 1
            FileList.append(os.path.join(path, file))
print('Number of Available Files: ' + str(NSample))
print(SampleList)
print(FileList)


def mkdir(my_path):
    if not os.path.exists(my_path):
        os.makedirs(my_path)


# The First
'''
TileSize = 512
slide = openslide.OpenSlide(FileList[0])
data_gen = DeepZoomGenerator(slide, tile_size=TileSize, overlap=0, limit_bounds=False)
addr = data_gen.level_tiles[data_gen.level_count - 1]
num_w = addr[0]
num_h = addr[1]
print('levels =', data_gen.level_count)
for i in range(num_w):
    for j in range(num_h):
        img = np.array(data_gen.get_tile(data_gen.level_count - 1, (i, j)))
        grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        ret, binImg = cv2.threshold(grey, 255*0.7, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
        if sum(sum(binImg)) > np.shape(binImg)[0] * np.shape(binImg)[1] * 0.2:
            if np.shape(binImg)[0] == TileSize and np.shape(binImg)[1] == TileSize:
                mkdir(os.path.join(TilePath, SampleList[0]))
                cv2.imwrite(os.path.join(TilePath, SampleList[0]) + '/Tile_' + str(i) + '_' + str(j) + '.tiff', img)
print('Done!\n')'''

TileSize = 512
for n in range(NSample):
    slide = openslide.OpenSlide(FileList[n])
    data_gen = DeepZoomGenerator(slide, tile_size=TileSize, overlap=0, limit_bounds=False)
    addr = data_gen.level_tiles[data_gen.level_count - 1]
    num_w = addr[0]
    num_h = addr[1]
    for i in range(num_w):
        for j in range(num_h):
            img = np.array(data_gen.get_tile(data_gen.level_count - 1, (i, j)))
            grey = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            ret, binImg = cv2.threshold(grey, 255 * 0.7, 255, cv2.THRESH_BINARY, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
            if sum(sum(binImg)) > np.shape(binImg)[0] * np.shape(binImg)[1] * 0.2:
                if np.shape(binImg)[0] == TileSize and np.shape(binImg)[1] == TileSize:
                    mkdir(os.path.join(TilePath, SampleList[n]))
                    cv2.imwrite(os.path.join(TilePath, SampleList[n]) + '/Tile_' + str(i) + '_' + str(j) + '.jpeg', img)
                    # lossy format is acceptable
    print(str(n+1) + ' of ' + str(NSample) + ' finished')
print('Done!\n')
