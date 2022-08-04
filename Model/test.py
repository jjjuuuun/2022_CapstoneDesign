import os
from CycleGAN.options.test_options import TestOptions
from CycleGAN.models import create_model
import cv2
import time
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
from YOLO_V3.face_detaction import get_detected_img
from Poisson_Image_Editing.create_mask import *
from Poisson_Image_Editing.seamless_cloning import *
from tqdm import tqdm

if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    detect_path = opt.yolo_path
    #dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # if not os.path.exists(f'./result_img/{opt.name}/train{opt.webtoon_direction}'):
    #     os.makedirs(f'./result_img/{opt.name}/train{opt.webtoon_direction}')
    if opt.webtoon_direction == 'A':
        save_path = os.path.join(f'./result_img/{opt.name}/{opt.webtoon_direction}B')
    else:
        save_path = os.path.join(f'./result_img/{opt.name}/A{opt.webtoon_direction}')
    # Detection
    cv_net_yolo = cv2.dnn.readNetFromDarknet(os.path.join(detect_path, 'yolov3-face.cfg'), os.path.join(detect_path, 'yolov3-wider_16000.weights'))
    conf_threshold = 0.5
    nms_threshold = 0.4

    files = os.listdir(f'{opt.dataroot}/train{opt.webtoon_direction}')
    for cut in tqdm(files):
        original_img = cv2.imread(f'{opt.dataroot}/train{opt.webtoon_direction}/{cut}')
        img = original_img.copy()
        draw_img, face, face_index = get_detected_img(cv_net_yolo, img, conf_threshold=conf_threshold, nms_threshold=nms_threshold, use_copied_array=True, is_print=False)
        if len(face_index)==0:
            pass
        else:
            source = np.zeros_like(original_img)
            mask = source.copy()
            mask = cv2.cvtColor(mask,cv2.COLOR_BGR2GRAY)
            if opt.eval:
                model.eval()
            for i, data in enumerate(face):
                height, widht = data.shape[:2]
                data2 = cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(data2).convert('RGB')
                transform_list = []
                osize = [256, 256]
                transform_list.append(transforms.Resize(osize, Image.BICUBIC))
                transform_list.append(transforms.RandomCrop(256))
                transform_list += [transforms.ToTensor()]
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
                transform = transforms.Compose(transform_list)
                
                if i >= opt.num_test:  # only apply our model to opt.num_test images.
                    break
                if opt.webtoon_direction == 'A':
                    A = transform(img)
                    model.real_A = A
                    model.real_B = A
                    fake = 'B'
                else:
                    B = transform(img)
                    model.real_A = B
                    model.real_B = B
                    fake = 'A'

                model.test()
                visuals = model.get_current_visuals()  # get image results
                img_path = model.get_image_paths()     # get image paths
                #if i % 5 == 0:
                    #print('processing (%04d)-th image... %s' % (i, img_path))
                    
                result = visuals[f'fake_{fake}']
                result = result.cpu().float().numpy()
                result = (np.transpose(result, (1, 2, 0)) + 1) / 2.0 * 255.0
                result = result.astype(np.uint8)
                transformed_img = cv2.resize(result, dsize=(widht,height), interpolation=cv2.INTER_AREA)
                mask, source = create_mask(source,mask,transformed_img,face_index[i])

            source,mask,target = convert_img(source,mask,original_img)
            cloner = PoissonSeamlessCloner(mask,source,target,"spsolve", 1.0)


            img = cloner.poisson_blend_rgb("alpha", 1.0)
            
            img = (img * 255).astype(np.uint8)
            Image.fromarray(img).save(os.path.join(save_path, cut))