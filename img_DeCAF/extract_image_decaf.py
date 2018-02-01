import sys

sys.path.append('/home/niuli/extract_feature/decaf_release/')

from decaf.scripts import imagenet
from collections import defaultdict
import cv2
import time
import numpy as np
import tarfile
import tempfile
import os
import gzip
import shutil

data_root = '/home/niuli/extract_feature/decaf_release/data/decaf_pretrained/'
net = imagenet.DecafNet(data_root+'imagenet.decafnet.epoch90', data_root+'imagenet.decafnet.meta')

imagedir = '/home/niuli/niuli_passport2/Flickr_Hollywood2/crawl_web_data/crawl_Flickr_image/downloads'
imagelistfile = 'Flickr_image_list_linux.txt'
decaf_dir = '/home/niuli/niuli_passport2/Flickr_Hollywood2/extracted_features/Flickr_image_decaf'

if not os.path.exists(decaf_dir):
    os.makedirs(decaf_dir)

try:
    imagelist= file(imagelistfile).read().split('\n')[:-1]
except IOError:
    logging.error('%s does not exists', imagelistfile)

def extract_batch(imagelist):
	for i, member in enumerate(imagelist):
            if not member:
                continue
	    imgfile = os.path.join(imagedir, member)
	    #print imgfile
	    feat_save_file =  os.path.join(decaf_dir, member.replace('.jpg', '.fc6')) 
	    feat_save_file2 =  os.path.join(decaf_dir, member.replace('.jpg', '.fc7')) 
	    if os.path.exists(feat_save_file):
		continue

            print imgfile
	    img = cv2.imread(imgfile)
	    if img is None:
                continue
            try:
                img = cv2.resize(img, dsize=(256,256))
            except:
                continue
	    scores = net.classify(img, center_only=True)
	    feat = net.feature('fc6_neuron_cudanet_out')[0]    
	    feat2 = net.feature('fc7_neuron_cudanet_out')[0]    
	    if not os.path.exists(os.path.dirname(feat_save_file)):
		os.makedirs(os.path.dirname(feat_save_file))
	    np.savetxt(feat_save_file, feat, fmt='%.10g')
	    np.savetxt(feat_save_file2, feat2, fmt='%.10g')

if __name__=='__main__':
    extract_batch(imagelist)
