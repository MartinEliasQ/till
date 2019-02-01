"""Source :https://colab.research.google.com/drive/109vu3F1LTzD1gdVV6cho9fKGx7lzbFll#scrollTo=8q8a2Ha9pnaz """

import matplotlib.pyplot as plt
import numpy as np
import torchvision
from imgaug import augmenters as iaa
import imgaug as ia


class ImgAugTransform:
    def __init__(self):

        self.aug = iaa.Sequential([
            iaa.Noop(),
            iaa.Scale((224, 224)),
            iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0))),
            iaa.Fliplr(0.5),
            iaa.Affine(rotate=(-20, 20), mode='symmetric'),
            iaa.Sometimes(0.25,
                          iaa.OneOf([iaa.Dropout(p=(0, 0.1)),
                                     iaa.CoarseDropout(0.1, size_percent=0.5)])),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True)
        ])

    @staticmethod
    def show_dataset(dataset, n=6):
        import matplotlib as mpl
        print(dataset)
        mpl.rcParams['axes.grid'] = False
        mpl.rcParams['image.interpolation'] = 'nearest'
        mpl.rcParams['figure.figsize'] = 15, 25
        img2 = np.vstack((np.hstack((np.asarray(dataset[i][0]) for _ in range(n)))
                          for i in range(len(dataset))))
  #      print(img)
   #     plt.imshow(img)
    #    plt.axis('off')

    def __call__(self, img):
        print("call")
        print(img)
        img = np.array(img)
        return self.aug.augment_image(img)
