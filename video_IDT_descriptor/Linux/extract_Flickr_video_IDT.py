#!/bin/python

import fnmatch;
import os;
import sys;
import shutil;
import re;
import time;
import threading;
import tempfile
from time import clock;

video_dir= "/home/niuli/niuli_passport2/Flickr_Hollywood2/crawl_web_data/crawl_Flickr_video/downloads";
feat_dir= "/home/niuli/niuli_passport2/Flickr_Hollywood2/extracted_features/Flickr_video_idt";
cmd= "/home/niuli/trajectory_release/improved_trajectory_release/release/DenseTrackStab";

nThreads = 5;

def get_filelist():
    files = file('flickr_video_list_linux.txt').read().split('\n');
    print len(files);
    return files;

def extract_one(video_file, feat_file, time_file):
    if not video_file:
        return;
    if os.path.exists(time_file):
    	return;
    f = open(time_file, 'w');
    cmdline = '%s \'%s\' -L 50 -W 16 -s 2 -t 3 > \'%s\'' % (cmd, video_file, feat_file);
    print cmdline;
    time_start_ = time.time();
    os.system(cmdline);
    time_end_ = time.time();    
    print>>f, '%g sec.' % (time_end_-time_start_,);
    f.close();
    print "finish_time:"+str(clock());

class myThread (threading.Thread):
	def __init__(self, threadID):
		self.threadID = threadID;
		threading.Thread.__init__(self);
	
	def run(self):
		threadLock.acquire();
		fname="";
		if len(FILENAMES)>0:
		    fname = FILENAMES.pop();
		    print "%d files remaining" % (len(FILENAMES));
		threadLock.release();
		if len(fname)>0:
                    tmp_fname = fname.split('/');
                    folder_name = tmp_fname[0];
                    base_name,ext = os.path.splitext(tmp_fname[1]);
		    input_file = os.path.join(video_dir, fname);
                    if not os.path.exists(os.path.join(feat_dir, folder_name)):
                        try:
                            os.makedirs(os.path.join(feat_dir,folder_name));
                        except:
                            pass;
		    output_file1 = os.path.join(feat_dir, folder_name, base_name+'.feat');
		    output_file2 = os.path.join(feat_dir, folder_name, base_name+'.time');
		    extract_one(input_file, output_file1, output_file2);


print "start_time:"+str(clock());

threadLock = threading.Lock();

FILENAMES = get_filelist();


threads = [];
for i in range(nThreads):
    threads.append(myThread(i));

for t in threads:
    t.start();


while len(FILENAMES)>0:
    for i in range(len(threads)):
	if not threads[i].isAlive():
	   threads[i]=myThread(i);
  	   threads[i].start();
print "Exiting Main Tread";

