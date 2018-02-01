import os
from glob import glob
import re
from optparse import OptionParser

tag_dir = 'H:\Flickr_Hollywood2\crawl_web_data\crawl_Flickr_video\downloads';
tf_dir = 'H:\Flickr_Hollywood2\extracted_features\Flickr_video_tf';

dim = 2000;
word_dict = dict();
in_file = open('word_freq.txt', 'r');
for (line_num, line) in enumerate(in_file):
    if (line_num == dim):
        break;
    word = line.split()[0];
    word_dict[word] = line_num;
in_file.close();

queries = file('queries.txt').read().split('\n');
for query in queries:
    tag_files = glob(os.path.join(tag_dir,query,'*.tag'))
    for tag_file in tag_files:
        line = file(tag_file).read();
            
        if not cmp(line,''):
            continue;

        save_file = os.path.join(tf_dir, query, os.path.basename(tag_file).replace('.tag','.tf'));
        print save_file
        if not os.path.exists(os.path.dirname(save_file)):
            os.makedirs(os.path.dirname(save_file))
     
        outfile = open(save_file, 'w')
        words = line.split()
        feature = [0] * dim
        for w in words:
            if w in word_dict:
                feature[word_dict[w]] += 1
        for di in range(0,dim):
            outfile.write(str(feature[di])+'\n')
        outfile.close()




