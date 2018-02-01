import os
from glob import glob
import re
from optparse import OptionParser
import nltk;
from nltk.corpus import wordnet as wn;

in_file = open('forbidden_words.txt', 'r');
forbid_words = set(in_file.read().splitlines());
in_file.close();

queries = file('queries.txt').read().split('\n')

freq_dict = dict();

# Bing image
tag_dirs = {'H:\Flickr_Hollywood2\crawl_web_data\crawl_Bing_image\downloads',
           'H:\Flickr_Hollywood2\crawl_web_data\crawl_Google_image\downloads',
           'H:\Flickr_Hollywood2\crawl_web_data\crawl_Flickr_video\downloads'}

for tag_dir in tag_dirs:
    for query in queries:
        if not query:
            continue
        file_list = glob(os.path.join(tag_dir, query, '*.tag'))
        for file_name in file_list:
            line = file(file_name).read();
            words = line.split();
            for w in words[0:]:
                if freq_dict.has_key(w):
                    freq_dict[w] += 1;
                else:
                    freq_dict[w] = 1;
             

freq_arr = freq_dict.items();
freq_arr.sort(lambda x, y: y[1] - x[1]);

outfile = open('word_freq.txt', 'w');
icount = 0;
for item in freq_arr:
    if len(item[0])>2 and not item[0] in forbid_words:
        pos_tag = nltk.pos_tag([item[0]])[0][1];
        if pos_tag in ['NN','JJ','VB']:
            print item[0];    
            outfile.write(item[0] + ' ' + str(item[1]) + '\n');
            icount += 1;
            if icount>=2000:
                break;

outfile.close();


