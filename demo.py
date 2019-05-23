from PIL import Image
import os
import numpy as np
from model2 import MultiTaskRecognizer
import matplotlib.pyplot as plt
import tqdm

if not os.path.exists('demo'):
    os.makedirs('demo')

standard_dims = (40, 40)

f_test = open('MAFL/testing.txt', 'r').readlines()
m = MultiTaskRecognizer()

for image in tqdm.tqdm(f_test, total=len(f_test)):
    image = image.strip()
    img = Image.open('MAFL/img_align_celeba/' + image)

    landmarks = m.predict(np.expand_dims(np.asarray(img.convert('L').resize(standard_dims)), axis=-1))['Landmarks']

    x = landmarks[0][:5]
    y = landmarks[0][5:]

    x = x * img.size[0] / standard_dims[0]
    y = y * img.size[1] / standard_dims[1]

    fig = plt.imshow(img)
    plt.plot(x, y, 'ro')
    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    fig.axes.get_xaxis().set_visible(False)
    fig.axes.get_yaxis().set_visible(False)

    plt.savefig('demo/' + image, bbox_inches='tight', pad_inches=0)
    plt.close()
