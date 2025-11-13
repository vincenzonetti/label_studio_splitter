#given the folder downloaded with the option "coco format with images" from label studio
#this script splits it into into train validation and test according to the format accepted
#YOLO from ultralytics and COCO from YOLOX

#It also allows you to filter the json file in case there are some classes in which you are not interested into
#/home/vincenzo/Desktop/Uni/thesis/unitn-polarized-camera/AI/Object_Detection/datasets/phone_detection/base_2k_img

import os 
import shutil
import argparse
from coco_filter import CocoFilter
import json
from pylabel import importer


def print_dataset_info(dataset):
    print(f"Number of images: {dataset.analyze.num_images}")
    print(f"Number of classes: {dataset.analyze.num_classes}")
    print(f"Classes:{dataset.analyze.classes}")
    print(f"Class counts:\n{dataset.analyze.class_counts}")
    print(f"Path to annotations:\n{dataset.path_to_annotations}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Filter COCO JSON: "
    "Filters a COCO Instances JSON file to only include specified categories. "
    "This includes images, and annotations. Does not modify 'info' or 'licenses'.")
    
    parser.add_argument("-f","--folder", dest="download_folder",
        help="path to the downloaded folder from label studio")
    parser.add_argument("-o", "--output_folder", dest="output_folder",
        help="path to the output folder in which the new split will be saved")
    
    parser.add_argument("-c", "--categories", nargs='+', dest="categories",
        help="List of category names to keep, separated by spaces, e.g. -c person dog bicycle")
    
    parser.add_argument("-s", "--split", nargs=3, dest="split",
        help="[Train, Val, Test] for splitting the dataset e.g. -s 75 10 15 ")
    
    
    args = parser.parse_args()
    cf = CocoFilter()
    download_folder = args.download_folder
    output_folder = args.output_folder
    #assume the naming is the one downloaded from label studio
    json_file = "result.json"
    img_dir = "images"
    category_names = args.categories[0].split()
    filtered_annotations = cf.main(input_json=os.path.join(download_folder,json_file),categories=category_names)

    #This passage is necessary since importer needs to a json file
    with open('tmp.json', 'w') as fp:
        json.dump(filtered_annotations, fp)
        
    path_to_images = os.path.join(download_folder,img_dir)
    dataset = importer.ImportCoco("tmp.json", path_to_images=path_to_images, name="phone_full_coco")
    os.remove("tmp.json")
    print_dataset_info(dataset)
    dataset.splitter.StratifiedGroupShuffleSplit(train_pct=int(args.split[0])/100, val_pct=int(args.split[1])/100, test_pct=int(args.split[2])/100, batch_size=1)
    dataset.analyze.ShowClassSplits()
    
    yolo_lbl_dir = os.path.join(output_folder,'yolo','labels')
    dataset.export.ExportToYoloV5(output_path=yolo_lbl_dir,yaml_file='dataset.yaml', copy_images=True, use_splits=True,cat_id_index=0)
    splits = ['train','test','val']
    coco_dir = os.path.join(output_folder,'coco')
    os.makedirs(coco_dir,exist_ok=True)
    coco_annotation_dir = os.path.join(coco_dir,'annotations')
    os.makedirs(coco_annotation_dir,exist_ok=True)
    
    for split in splits:
        output_tmp_dir = os.path.join(coco_dir,split)
        yolo_lbl_split_dir = os.path.join(yolo_lbl_dir,split)
        yolo_img_split_dir = os.path.join(output_folder,'yolo','images',split)
        dataset = importer.ImportYoloV5(path=yolo_lbl_split_dir, path_to_images=yolo_img_split_dir, cat_names=category_names, name="coco128")
        
        coco_split = split if split!="val" else "valid"
        dataset.export.ExportToCoco(output_path=f'{coco_annotation_dir}/{coco_split}.json',cat_id_index=0)

        coco_img_split_dir = os.path.join(coco_dir,f"{split}2017")
        
        shutil.copytree(yolo_img_split_dir, coco_img_split_dir)
    