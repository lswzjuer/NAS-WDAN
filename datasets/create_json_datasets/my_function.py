import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

def run_length_encoding(x):
    dots = np.where(x.T.flatten() == 1)[0]
    run_lengths = []
    prev = -2
    for b in dots:
        if (b > prev + 1): run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    run_lengths = ' '.join([str(r) for r in run_lengths])
    return run_lengths


def remove_duplicate(mask, threshold=0.7, scores=None):
    if scores is None:
        scores = np.sum(mask, axis=(0, 1))  ## Use size of nucleus as score...
    order = np.argsort(scores)[::-1] + 1  # 1-based descending
    flat_mask = np.max(mask * np.reshape(order, [1, 1, -1]), -1)
    plt.imshow(flat_mask)
    #plt.savefig('/media/szu/hardware/ycl/maskscoring_rcnn/demo/test.png')
    for i in np.arange(len(order)):
        mask[:, :,i] = mask[:, :,i] * (flat_mask == order[i])

    new_scores = np.sum(mask, axis=(0, 1))
    diff_pix = scores - new_scores
    reduccion = diff_pix / scores
    # if we have reduced particle size in more than xx percent, remove particle
    ## This only has some effect during the test time augmentation merge
    #mask[ :, :,reduccion > threshold] = 0
    return mask


def numpy2encoding(predicts, img_name, scores=None, threshold=0.85, dilation=False):

    ImageId = []
    EncodedPixels = []

    rle = run_length_encoding(predicts[:, :])

    if len(rle) > 0:
        ImageId.append(img_name)
        EncodedPixels.append(rle)
    return ImageId, EncodedPixels, predicts


def write2csv(file, ImageId, EncodedPixels):
    df = pd.DataFrame({'ImageId': ImageId, 'EncodedPixels': EncodedPixels})
    df.to_csv(file, index=False, columns=['ImageId', 'EncodedPixels'])



