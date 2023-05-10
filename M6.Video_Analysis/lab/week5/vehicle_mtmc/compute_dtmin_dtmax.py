import os
import cv2

path = '/ghome/group03/dataset/aic19-track1-mtmc-train/train/S04/'

f_of = open('/ghome/group03/dataset/aic19-track1-mtmc-train/cam_timestamp/S04.txt','r')
lines = f_of.readlines()
offsets = {}
for line in lines:
    cam,t = line.split(' ')
    offsets[cam] = float(t.split('\n')[0])
f_of.close()


f_f = open('/ghome/group03/dataset/aic19-track1-mtmc-train/cam_framenum/S04.txt','r')
lines = f_f.readlines()
frames = {}
for line in lines:
    cam,t = line.split(' ')
    frames[cam] = float(t.split('\n')[0])
f_f.close()

sum = 0
dtmax = []

for cam in os.listdir(path):
    video = cv2.VideoCapture(path+'/'+cam+'/vdo.avi')
    fps = video.get(cv2.CAP_PROP_FPS)
    duration = frames[cam]/fps

    sum += (1/fps)
    dtmax.append(duration-offsets[cam])


print("dtmin:", max(0,sum))
print("dtmax:", min(dtmax))
