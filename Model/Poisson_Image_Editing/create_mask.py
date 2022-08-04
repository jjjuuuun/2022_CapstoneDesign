import numpy as np
import cv2
from PIL import Image

def create_mask(source,mask,trans_img,idx):

    mask[idx[0]:idx[1],idx[2]:idx[3]] = [255]
    source[idx[0]:idx[1],idx[2]:idx[3]] = trans_img

    return mask,source

def convert_img(source,mask,original_img):
    source = Image.fromarray(source).convert('RGB')
    source = np.array(source)
    source = source.astype(np.float64) / 255
    mask = Image.fromarray(mask).convert('L')
    mask = np.array(mask)
    mask = mask.astype(np.float64) / 255
    target = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    target = Image.fromarray(target).convert('RGB')
    target = np.array(target)
    target = target.astype(np.float64) / 255

    return source, mask, target