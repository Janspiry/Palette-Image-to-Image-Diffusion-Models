import os
import numpy as np
from sklearn.model_selection import train_test_split
import cv2

def convert_abl(ab, l):
    """ convert AB and L to RGB """
    l = np.expand_dims(l, axis=3)
    lab = np.concatenate([l, ab], axis=3)
    if len(lab.shape)==4:
        image_color, image_l = [], []        
        for _color, _l in zip(lab, l):
            out = cv2.cvtColor(_color.astype('uint8'), cv2.COLOR_LAB2RGB) 
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR) 
            image_color.append(out)
            image_l.append(cv2.cvtColor(_l.astype('uint8'), cv2.COLOR_GRAY2RGB))
        image_color = np.array(image_color)
        image_l = np.array(image_l)
    else:
        image_color = cv2.cvtColor(lab.astype('uint8'), cv2.COLOR_LAB2RGB)
        image_l = cv2.cvtColor(l.astype('uint8'), cv2.COLOR_GRAY2RGB)
    return image_color, image_l

def load_data(home):
    ab1 = np.load(os.path.join(home,"ab/ab", "ab1.npy"))
    ab2 = np.load(os.path.join(home, "ab/ab", "ab2.npy"))
    ab3 = np.load(os.path.join(home,"ab/ab", "ab3.npy"))
    ab = np.concatenate([ab1, ab2, ab3], axis=0)
    l = np.load(os.path.join(home,"l/gray_scale.npy"))
    return ab, l

if __name__ == '__main__':
    home = './' # path saved .npy
    flist_save_path = './flist'
    image_save_path = './images' # images save path

    all_color, all_l = load_data(home)
    image_color, image_l = convert_abl(all_color, all_l)
    
    color_save_path, gray_save_path  = '{}/color'.format(image_save_path), '{}/gray'.format(image_save_path)
    os.makedirs(color_save_path, exist_ok=True)
    os.makedirs(gray_save_path, exist_ok=True)
    for i in range(image_color.shape[0]):
        cv2.imwrite('{}/{}.png'.format(color_save_path, str(i).zfill(5)), image_color[i])
    for i in range(image_l.shape[0]):
        cv2.imwrite('{}/{}.png'.format(gray_save_path, str(i).zfill(5)), image_l[i])
    
    os.makedirs(flist_save_path, exist_ok=True)
    arr = np.random.permutation(25000)
    with open('{}/train.flist'.format(flist_save_path), 'w') as f:
        for item in arr[:24000]:
            print(str(item).zfill(5), file=f)
    with open('{}/test.flist'.format(flist_save_path), 'w') as f:
        for item in arr[24000:]:
            print(str(item).zfill(5), file=f)