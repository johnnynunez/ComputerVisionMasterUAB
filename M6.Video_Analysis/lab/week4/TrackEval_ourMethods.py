import argparse
import os
import pickle

import pandas as pd


def preprocess_data(trackings_predicted, trackings_gt):
    """Preprocess the data from the tracker
    Delete the initial frames which are not in the ground truth

    Args:
        trackings_predicted (str): path to the predicted trackings
        trackings_gt (str): path to the ground truth trackings
    Returns:
        data_pred_new (list): list of the lines of the predicted trackings that are in the ground truth
    """
    # Load .txt files
    with open(trackings_predicted, "r") as f:
        data_pred = f.readlines()
    with open(trackings_gt, "r") as f:
        data_gt = f.readlines()
    
    # Read the first number of each line of the gt file
    for i in range(len(data_gt)):
        data_gt[i] = data_gt[i].split(",")[0]
        
    # Only keep the lines of the predicted file that have the same number as the first number of the gt file
    data_pred_new = []
    for i in range(len(data_pred)):
        data_pred_frame = data_pred[i].split(",")[0]
        if data_pred_frame in data_gt:
            data_pred_new.append(data_pred[i])
            
    return data_pred_new


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(
        description="evaluate the tracking")
    parser.add_argument("--seq", type=str, default = "S03")
    parser.add_argument("--csv_name", type=str, default = "w4_OF_S03.csv")
    parser.add_argument("--input", type=str, default="/ghome/group03/mcv-m6-2023-team6/week4/Results/Task2/")
    parser.add_argument("--output", type=str, default="/ghome/group03/mcv-m6-2023-team6/week4/TrackEval/S03/data/trackers/mot_challenge/MOT17-train/MPNTrack/")
    parser.add_argument("--evaluated", type=int, default = 0)
    parser.add_argument("--evalOutput", type=str, default = "/ghome/group03/mcv-m6-2023-team6/week5/Results/TrackEvalResults")
    parser.add_argument("--gt_path_1", type=str, default = "/ghome/group03/mcv-m6-2023-team6/week5/TrackEval")
    parser.add_argument("--gt_path_2", type=str, default = "data/gt/mot_challenge/MOT17-train")
    args = parser.parse_args()


    eval = bool(args.evaluated)

    
    if not eval:
        print("Evaluating the tracking")
         
        for file in os.listdir(args.input):
            if file.endswith(".txt") :
            
                if os.path.exists(os.path.join(args.output + "data/", file.split(".")[0] + ".txt")):
                    os.remove(os.path.join(args.output + "data/", file.split(".")[0] + ".txt"))
                        
                    path_gt = os.path.join(args.gt_path_1, args.seq, args.gt_path_2, file.split(".")[0], "gt", "gt.txt")

                    data = preprocess_data(os.path.join(args.input,file) , path_gt)
                    # save the data in .txt format
                    with open(os.path.join(args.output + "data/", file.split(".")[0] + ".txt"), "w") as f:
                        for line in data:
                            f.write(line)
                     
        
    else:
        print("Moving the csv")
        # copy the generated csv file to the folder 
        #get the name of the csv file in the output folder
        files = os.listdir(args.output)
        for file in files:
            if file.endswith(".csv"):
                #open the file and convert it to a dataframe
                dataframe = pd.read_csv(os.path.join(args.output, file))

                # create a new dataframe only with the columns we want: HOTA,IDF1, IDP,IDR, DetPr___AUC,DetRe___AUC
                new_dataframe = dataframe[['HOTA(0)','HOTA___AUC','IDF1', 'IDP','IDR', 'DetPr___AUC','DetRe___AUC']]
                #save the new dataframe to a csv file
                new_dataframe.to_csv(os.path.join(args.evalOutput, args.csv_name), index=False)
                print(os.path.join(args.evalOutput, args.csv_name))
                #delete the file
                os.remove(os.path.join(args.output, file))
                



    
