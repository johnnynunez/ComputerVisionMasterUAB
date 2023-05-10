from util import *
import os

def create_dataset(path):

    if not os.path.exists(path+'/labels'):
        os.makedirs(path+'/labels')

    frames = len(os.listdir(path+'/frames/'))

    gt = load_from_txt(path+'/gt/gt.txt')

    for f in range(1,frames+1):
        if f in gt:
            file = open(path+'/labels/'+str(f)+'.txt','w')
            for det in gt[f]:
                line = f'{det[0]} {det[1]} {det[2]} {det[3]} {det[4]} \n'
                file.write(line)

            file.close()


seqs = ['S01','S04']
path = '/export/home/group03/dataset/aic19-track1-mtmc-train/train/'



for seq in seqs:
    for c in os.listdir(path+'/'+seq):
        create_dataset(path+'/'+seq+'/'+c)
