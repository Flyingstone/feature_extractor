import os
from glob import glob
import re
from nltk.corpus import wordnet as wn
from optparse import OptionParser

src_dir = 'H:\Flickr_Hollywood2\crawl_web_data\crawl_Google_image\downloads';
tgt_dir = 'H:\Flickr_Hollywood2\crawl_web_data\crawl_Google_image\downloads';

queries = file('queries.txt').read().split('\n');

short_len = 2;
long_len  = 16


def morph(w):
    for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]:
        orig = wn.morphy(w, pos);
        if (orig != None):
            return orig;
    return w

# build the forbidden word set
in_file = open('forbidden_words.txt', 'r');
forbid_words = set(in_file.read().splitlines());
in_file.close();

# build the forbidden regular expression
in_file  = open('forbidden_patterns.txt', 'r');
raw_data = re.sub('\n{2,}', '\n', in_file.read());
forbid_re = '(' + '|'.join(raw_data.splitlines()) + ')';
forbid_am = re.compile(forbid_re);
in_file.close()


def extract_batch():

    for query in queries: 
        txt_file_list = glob(os.path.join(src_dir,query,'*.txt'));    
        for txt_file in txt_file_list:
            print txt_file
            [basename,ext] = os.path.splitext(os.path.basename(txt_file));
        
            tag_file =  os.path.join(tgt_dir, query, basename+'.tag');

            tag = file(txt_file).read();                    
            tag = re.sub(r'\\n',' ',tag);
            tag = re.sub(r'\\"','"',tag);
        
            tag = re.sub(r'\s+[^\s]*http:[^\s]*\s+',' ',tag);  
            tag = re.sub(r'\s+[^\s]*https:[^\s]*\s+',' ',tag);                          
            tag = re.sub(r'[,.:;\"\']',' ',tag);
            tag = re.sub(r'\s+[^\s]*[^a-zA-Z\s]+[^\s]*\s+',' ',tag);
            
            outfile = open(tag_file, 'w')
            line  = tag.decode('utf-8').lower();
            words = line.split();

            for w in words[0:]:
                w_len = len(w);
                if w.isalpha() and w_len > short_len and w_len < long_len and not w in forbid_words:
                    res = forbid_am.search(w);
                    if res == None:
                        outfile.write(morph(w).encode('utf-8')+' ');
            outfile.close()
            
    

if __name__=='__main__':
    extract_batch()
