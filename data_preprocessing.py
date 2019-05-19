import numpy as np
from PIL import Image

def global_epoch(file_path, update=None):
    if not update:
        try:
            f = open(file_path, 'r', encoding='utf8')
            val = int(f.read())
            f.close()
            return val
        except:
            f = open(file_path, 'w', encoding='utf8')
            f.write(str(0))
            f.close()
            return 0
    else:
        f = open(file_path, 'w', encoding='utf8')
        f.write(str(update))
        f.close()

def load_image(img_dir, shape, mode='L'):
    img = Image.open(img_dir).convert(mode)
    size = img.size
    img = img.resize(shape, Image.LANCZOS)
    img = np.asarray(img)
    img = img.astype('float32')
    '''img = img / 255.0
    img = img - 0.5
    img = img * 2.0'''
    if mode == 'L':
        img = np.expand_dims(img, axis=-1)
    return img, size