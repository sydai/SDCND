from PIL import Image
import os, numpy as np
folder = 'test_images'

read = lambda imname: np.asarray(Image.open(imname).convert("RGB"))

ims = [read(os.path.join(folder, filename)) for filename in os.listdir(folder)]
im_array = np.array(ims, dtype='uint8')

