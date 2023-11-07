import glob
import json
import copy
import os
import pandas as pd
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import torch

from skimage.segmentation import slic, find_boundaries
from scipy.ndimage import binary_dilation
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import warnings
warnings.filterwarnings("ignore")


class GPT4V(object):

    def __init__(self, cfg):
        self.cfg = cfg

        if 'sam' in self.cfg.region_division_methods:
            self.sam = sam_model_registry['vit_h'](checkpoint='pretrain/sam_vit_h_4b8939.pth')
            self.sam.to(device=self.cfg.device)
            self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def region_division(self):
        image_files = []
        if self.cfg.dataset_name in ['mvtec']:
            root = self.cfg.dataset_name
            image_files = glob.glob(f'{root}/*/test/*/???.png')
            # image_files = [image_file.replace(f'{root}/', '') for image_file in image_files]
        elif self.cfg.dataset_name in ['visa']:
            root = self.cfg.dataset_name
            CLSNAMES = [
                'pcb1', 'pcb2', 'pcb3', 'pcb4',
                'macaroni1', 'macaroni2', 'capsules', 'candle',
                'cashew', 'chewinggum', 'fryum', 'pipe_fryum',
            ]
            csv_data = pd.read_csv(f'{root}/split_csv/1cls.csv', header=0)
            columns = csv_data.columns  # [object, split, label, image, mask]
            test_data = csv_data[csv_data[columns[1]] == 'test']
            for cls_name in CLSNAMES:
                cls_data = test_data[test_data[columns[0]] == cls_name]
                cls_data.index = list(range(len(cls_data)))
                for idx in range(cls_data.shape[0]):
                    data = cls_data.loc[idx]
                    image_files.append(data[3])
            image_files = [f'{root}/{image_file}' for image_file in image_files]
        if len(image_files) == 0:
            return -1

        image_files.sort()
        for idx1, image_file in enumerate(image_files):
            self.img_size = self.cfg.img_size
            self.div_num = self.cfg.div_num
            self.div_size = self.cfg.img_size // self.cfg.div_num
            self.edge_pixel = self.cfg.edge_pixel
            img = cv2.imread(image_file)
            img = cv2.resize(img, (self.cfg.img_size, self.cfg.img_size))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            H, W, _ = img.shape

            if 'grid' in self.cfg.region_division_methods:
                masks = []
                for i in range(self.div_num):
                    for j in range(self.div_num):
                        mask = np.zeros((self.img_size, self.img_size), dtype=np.bool)
                        x1, x2 = j * self.div_size, (j + 1) * self.div_size
                        y1, y2 = i * self.div_size, (i + 1) * self.div_size
                        mask[y1:y2, x1:x2] = True
                        masks.append(mask)
                self.sovle_masks(img, image_file, masks, 'grid')
                print(f'{self.cfg.dataset_name} --> {idx1 + 1}/{len(image_files)} {image_file} for grid')
            if 'superpixel' in self.cfg.region_division_methods:
                regions = slic(img, n_segments=60, compactness=20)
                masks = []
                for label in range(regions.max() + 1):
                    mask = (regions == label)
                    masks.append(mask)
                self.sovle_masks(img, image_file, masks, 'superpixel')
                print(f'{self.cfg.dataset_name} --> {idx1 + 1}/{len(image_files)} {image_file} for superpixel')
            if 'sam' in self.cfg.region_division_methods:
                masks = self.mask_generator.generate(img)
                masks = [mask['segmentation'] for mask in masks]
                self.sovle_masks(img, image_file, masks, 'sam')
                print(f'{self.cfg.dataset_name} --> {idx1 + 1}/{len(image_files)} {image_file} for sam')

    def sovle_masks(self, img, image_file, masks, method):
        mask_edge = np.zeros_like(img, dtype=np.bool).astype(np.uint8) * 255
        mask_edge_number = np.zeros_like(img, dtype=np.uint8)
        img_edge = copy.deepcopy(img)
        img_edge_number = copy.deepcopy(img)

        masks = [mask for mask in masks if 600 < mask.sum() < 120000]
        for idx, mask in enumerate(masks):
            y_idx, x_idx = np.where(mask)

            center_y = y_idx.mean()
            center_x = x_idx.mean()
            distances_squared = (y_idx - center_y) ** 2 + (x_idx - center_x) ** 2
            min_index = np.argmin(distances_squared)
            xc, yc = x_idx[min_index], y_idx[min_index]

            mask1 = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
            boundaries = find_boundaries(mask1, mode='inner')
            boundaries = boundaries[1:-1, 1:-1]
            if self.edge_pixel > 1:
                boundaries = binary_dilation(boundaries, iterations=self.edge_pixel - 1)
            mask_edge[boundaries == True] = 255

            text = str(idx + 1)
            img_pil = Image.fromarray(mask_edge_number)
            draw = ImageDraw.Draw(img_pil)
            font = ImageFont.truetype('assets/arial.ttf', 10)
            text_width, text_height = draw.textsize(text, font)

            x1_bg, y1_bg = xc - text_width // 2, yc - text_height // 2,
            x2_bg, y2_bg = x1_bg + text_width, y1_bg + text_height
            draw.rectangle([(x1_bg, y1_bg + 2), (x2_bg, y2_bg)], fill=(128, 128, 128))
            draw.text((x1_bg, y1_bg), text, font=font, fill=(255, 255, 255))
            mask_edge_number = np.array(img_pil)
            mask_edge_number = np.maximum(mask_edge_number, mask_edge)

            img_edge[mask_edge > 100] = mask_edge[mask_edge > 100]
            img_edge_number[mask_edge_number > 100] = mask_edge_number[mask_edge_number > 100]

        suffix = os.path.splitext(image_file)[-1]
        cv2.imwrite(image_file.replace(suffix, f'_{method}_mask_edge.png'), cv2.cvtColor(mask_edge, cv2.COLOR_BGR2RGB))
        cv2.imwrite(image_file.replace(suffix, f'_{method}_mask_edge_number.png'), cv2.cvtColor(mask_edge_number, cv2.COLOR_BGR2RGB))
        cv2.imwrite(image_file.replace(suffix, f'_{method}_img_edge.png'), cv2.cvtColor(img_edge, cv2.COLOR_BGR2RGB))
        cv2.imwrite(image_file.replace(suffix, f'_{method}_img_edge_number.png'), cv2.cvtColor(img_edge_number, cv2.COLOR_BGR2RGB))
        torch.save(dict(masks=masks), image_file.replace(suffix, f'_{method}_masks.pth'))

        image_file_tmp = image_file
        if 'mvtec' in image_file_tmp:   # MVTec
            image_file_tmp = image_file_tmp.replace(image_file_tmp.split('/')[0], '')
        if 'visa' in image_file_tmp:  # VisA
            image_file_tmp = image_file_tmp.replace(image_file_tmp.split('/')[0], '')
        if image_file_tmp.startswith('/'):
            image_file_tmp = image_file_tmp[1:]
        cls_name = image_file_tmp.split('/')[0]
        number_list = [str(n) for n in list(range(10))]
        if cls_name[-1] in number_list:
            cls_name = cls_name[:-1]
        prompt_cls = f"This is an image of {cls_name}."
        prompt_describe = f"The image has different region divisions, each distinguished by white edges and each with a unique numerical identifier within the region, starting from 1. Each region may exhibit anomalies of unknown types, and if any region exhibits an anomaly, the normal image is considered anomalous. Anomaly scores range from 0 to 1, with higher values indicating a higher probability of an anomaly. Please output the image anomaly score, as well as the anomaly scores for the regions with anomalies. Provide the answer in the following format: \"image anomaly score: 0.9; region 1: 0.9; region 3: 0.7.\". Ignore the region that does not contain anomalies."
        f = open(image_file.replace(suffix, f'_prompt_wo_cls.txt'), 'w')
        f.write(f'{prompt_describe}')
        f.close()

        f = open(image_file.replace(suffix, f'_prompt.txt'), 'w')
        f.write(f'{prompt_cls} {prompt_describe}')
        f.close()

        f = open(image_file.replace(suffix, f'_{method}_out.txt'), 'w')
        f.write(f'')
        f.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu')

    # generate region divisions with labeled number
    parser.add_argument('--dataset_name', type=str, default='mvtec')
    # parser.add_argument('--dataset_name', type=str, default='visa')
    parser.add_argument('--region_division_methods', type=list, default=['superpixel'])
    # parser.add_argument('--region_division_methods', type=list, default=['grid', 'superpixel', 'sam'])
    parser.add_argument('--img_size', type=int, default=768)
    parser.add_argument('--div_num', type=int, default=16)
    parser.add_argument('--edge_pixel', type=int, default=1)

    cfg = parser.parse_args()
    runner = GPT4V(cfg)
    runner.region_division()
