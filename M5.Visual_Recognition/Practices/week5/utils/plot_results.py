import matplotlib.pyplot as plt
import os
import pandas as pd
import ast
import seaborn as sns
import numpy as np

task = 'task_a'
path_dir = '/ghome/group03/M5-Project/week5/results_new/task_a_new'

files = os.listdir(path_dir)



rows = []

for file in files:
    parts = file.split('_')
    
    text_emb= parts[3]
    image_emb = parts[2]
    margin = parts[9]
    
    if file == 'task_b_fasterRCNN_BERT_dim_out_fc_2048_margin_1.0_lr_0.0001':
        file = 'task_b_fasterRCNN_BERT_dim_out_fc_2048_margin_0.1_lr_0.0001'
    
    results_100_path = os.path.join(path_dir, file, 'test_100_after', 'results.txt')
    results_500_path = os.path.join(path_dir, file, 'test_500_after', 'results.txt')
    results_1000_path = os.path.join(path_dir, file, 'test_1000_after', 'results.txt')
    
    # Read the results from the files
    with open(results_100_path, 'r') as f:
        lines = f.readlines()
        lines = lines[0]
        lines = ast.literal_eval(lines)
        prec1_100 = float(lines[0].split()[-1])
        prec3_100 = float(lines[1].split()[-1])
        prec5_100 = float(lines[2].split()[-1])
        map_100 = float(lines[3].split()[-1])
        top1_100 = float(lines[4].split()[-1])
        top3_100 = float(lines[5].split()[-1])
        top5_100 = float(lines[6].split()[-1])
        
    with open(results_500_path, 'r') as f:
        lines = f.readlines()
        lines = lines[0]
        lines = ast.literal_eval(lines)
        prec1_500 = float(lines[0].split()[-1])
        prec3_500 = float(lines[1].split()[-1])
        prec5_500 = float(lines[2].split()[-1])
        map_500 = float(lines[3].split()[-1])
        top1_500 = float(lines[4].split()[-1])
        top3_500 = float(lines[5].split()[-1])
        top5_500 = float(lines[6].split()[-1])
    
    with open(results_1000_path, 'r') as f:
        lines = f.readlines()
        lines = lines[0]
        lines = ast.literal_eval(lines)
        prec1_1000 = float(lines[0].split()[-1])
        prec3_1000 = float(lines[1].split()[-1])
        prec5_1000 = float(lines[2].split()[-1])
        map_1000 = float(lines[3].split()[-1])
        top1_1000 = float(lines[4].split()[-1])
        top3_1000 = float(lines[5].split()[-1])
        top5_1000 = float(lines[6].split()[-1])
        
    # Add the results to the list of rows
    rows.append({
        'Samples': 100,
        'Text Embedding': text_emb,
        'Image Embedding': image_emb,
        'Margin': margin,
        'mAP': map_100,
        'Prec@1': prec1_100,
        'Prec@3': prec3_100,
        'Prec@5': prec5_100,
        'Top-1': top1_100,
        'Top-3': top3_100,
        'Top-5': top5_100
    })

    rows.append({
        'Samples': 500,
        'Text Embedding': text_emb,
        'Image Embedding': image_emb,
        'Margin': margin,
        'mAP': map_500,
        'Prec@1': prec1_500,
        'Prec@3': prec3_500,
        'Prec@5': prec5_500,
        'Top-1': top1_500,
        'Top-3': top3_500,
        'Top-5': top5_500
    })

    rows.append({
        'Samples': 1000,
        'Text Embedding': text_emb,
        'Image Embedding': image_emb,
        'Margin': margin,
        'mAP': map_1000,
        'Prec@1': prec1_1000,
        'Prec@3': prec3_1000,
        'Prec@5': prec5_500,
        'Top-1': top1_1000,
        'Top-3': top3_1000,
        'Top-5': top5_1000
    })
        
df = pd.DataFrame(rows, columns=['Samples', 'Text Embedding', 'Image Embedding', 'Margin', 'mAP', 'Prec@1', 'Prec@3', 'Prec@5', 'Top-1', 'Top-3', 'Top-5'])


# Plot a .csv file with the results
df.to_csv(f'/ghome/group03/M5-Project/week5/results/results_{task}.csv', index=False)


margins = [0.1, 1]

# create a new DataFrame with only the data that you want to plot
data = df[['Samples', 'Text Embedding', 'Image Embedding', 'Margin', 'mAP']].copy()
samples = [100, 500, 1000]

for margin in margins:
    plot_data = data[data['Margin'] == margin].copy()

    # Plot the results of map (y-axis) for the fasterRCNN + FastText, RESNET50 + FastText, fasterRCNN + BERT, RESNET50 + FastText  for 100, 500 and 1000 samples on x-axis
    # I want 4 bars for each bin (one for each model)
    fig, ax1 = plt.subplots(figsize=(8, 5))

    bins = np.arange(len(samples))
    width = 0.2

    for bin in bins:
        # fasterRCNN + FastText
        ax1.bar(bin - 3 * width / 2, plot_data['mAP'][plot_data['Image Embedding'] == 'fasterRCNN'][plot_data['Text Embedding'] == 'FastText'], width=width, alpha=0.5, color='#1f77b4', label='fasterRCNN + FastText')
        # RESNET50 + FastText
        ax1.bar(bin - width / 2, plot_data['mAP'][plot_data['Image Embedding'] == 'RESNET50'][plot_data['Text Embedding'] == 'FastText'], width=width, alpha=0.5, color='#ff7f0e', label='RESNET50 + FastText')
        # fasterRCNN + BERT
        ax1.bar(bin + width / 2, plot_data['mAP'][plot_data['Image Embedding'] == 'fasterRCNN'][plot_data['Text Embedding'] == 'BERT'], width=width, alpha=0.5, color='#2ca02c', label='fasterRCNN + BERT')
        # RESNET50 + BERT
        ax1.bar(bin + 3 * width / 2, plot_data['mAP'][plot_data['Image Embedding'] == 'RESNET50'][plot_data['Text Embedding'] == 'BERT'], width=width, alpha=0.5, color='#d62728', label='RESNET50 + BERT')

    ax1.set_ylabel('mAP')
    ax1.set_xlabel('Images considered')
    ax1.set_xticks(bins)

    # fig.tight_layout()
    fig.savefig(f'/ghome/group03/M5-Project/week5/results/histogram_map_{task}_margin_{margin}.png')



    
    
    
    
    