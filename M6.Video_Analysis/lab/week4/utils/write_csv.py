import pickle

import pandas as pd

images = pickle.load(open('/Users/AnaHarris/Documents/MASTER/M6/project/lab4/tracking_maskflownet.pkl', 'rb'))
df_list = []
for frame_id in images:
    for track in images[frame_id]:
        width = track[3] - track[1]
        height = track[4] - track[2]
        bb_left = track[1]
        bb_top = track[2]
        df_list.append(pd.DataFrame({'frame': int(frame_id), 'id': int(track[-1]), 'bb_left': bb_left, 'bb_top': bb_top,
                                     'bb_width': width, 'bb_height': height, 'conf': track[-2], "x": -1, "y": -1,
                                     "z": -1}, index=[0]))

df = pd.concat(df_list, ignore_index=True)
df = df.sort_values(by=['frame'])
df['frame'] = df['frame'] + 1

df.to_csv(f'C:/Users/AnaHarris/Documents/MASTER/M6/project/lab4/maskflownet.csv', index=False)
