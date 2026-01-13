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


def ultralytics_compatibility(base_folder, yaml_filename):
    """
    Adjust yolo dir for ultralytics compatibility.
    
    Current structure:
    images/
        train/
        test/
        val/
    labels/
        train/
        test/
        val/
    data.yaml
    
    Goal structure:
    train/
        images/
        labels/
    test/
        images/
        labels/
    val/
        images/
        labels/
    data.yaml
    """
    
    splits = ['train', 'test', 'val']
    
    # Create temporary directory to avoid conflicts
    temp_dir = os.path.join(base_folder, 'temp_restructure')
    os.makedirs(temp_dir, exist_ok=True)
    
    # Move files to new structure in temp directory
    for split in splits:
        # Create split directory with subdirectories
        split_dir = os.path.join(temp_dir, split)
        os.makedirs(split_dir, exist_ok=True)
        
        split_images_dir = os.path.join(split_dir, 'images')
        split_labels_dir = os.path.join(split_dir, 'labels')
        
        # Source directories
        src_images = os.path.join(base_folder, 'images', split)
        src_labels = os.path.join(base_folder, 'labels', split)
        
        # Copy images and labels to new structure
        if os.path.exists(src_images):
            shutil.copytree(src_images, split_images_dir)
        if os.path.exists(src_labels):
            shutil.copytree(src_labels, split_labels_dir)
    
    # Copy yaml file to temp directory
    yaml_src = os.path.join(base_folder, yaml_filename)
    yaml_dst = os.path.join(temp_dir, yaml_filename)
    if os.path.exists(yaml_src):
        shutil.copy2(yaml_src, yaml_dst)
    
    # Remove old structure
    old_images_dir = os.path.join(base_folder, 'images')
    old_labels_dir = os.path.join(base_folder, 'labels')
    
    if os.path.exists(old_images_dir):
        shutil.rmtree(old_images_dir)
    if os.path.exists(old_labels_dir):
        shutil.rmtree(old_labels_dir)
    if os.path.exists(yaml_src):
        os.remove(yaml_src)
    
    # Move everything from temp directory to base folder
    for item in os.listdir(temp_dir):
        src = os.path.join(temp_dir, item)
        dst = os.path.join(base_folder, item)
        shutil.move(src, dst)
    
    # Remove temp directory
    shutil.rmtree(temp_dir)
    
    print(f"Restructured {base_folder} for Ultralytics compatibility")


def rf_detr_compatibility(coco_dir, rf_detr_dir):
    """
    RF-DETR folder needs to be organized as follows:
    train/
        _annotations.coco.json
        img_train01
        img_train02
    test/
        _annotations.coco.json
        img_test01
        img_test02
    valid/
        _annotations.coco.json
        img_valid01
        img_valid02
    
    The annotations for the respective folder have to be picked from the coco's directory
    e.g. _annotations.coco.json of the train folder can be found in coco_dir/annotations/train.json
    The same for the other splits.
    The images found also in the coco's directory e.g. for training copy the images in the folder coco_dir/train2017
    Remember to keep the 2017 in the naming of the split
    """
    
    # Define split mappings (coco split name -> rf_detr split name)
    split_mapping = {
        'train': 'train',
        'test': 'test',
        'val': 'valid'  # Note: val -> valid
    }
    
    for coco_split, rf_split in split_mapping.items():
        # Create split directory in rf_detr
        rf_split_dir = os.path.join(rf_detr_dir, rf_split)
        os.makedirs(rf_split_dir, exist_ok=True)
        
        # Copy annotation file
        # Source: coco_dir/annotations/{coco_split}.json (or valid.json for val)
        annotation_src_name = 'valid.json' if coco_split == 'val' else f'{coco_split}.json'
        annotation_src = os.path.join(coco_dir, 'annotations', annotation_src_name)
        annotation_dst = os.path.join(rf_split_dir, '_annotations.coco.json')
        
        if os.path.exists(annotation_src):
            shutil.copy2(annotation_src, annotation_dst)
            print(f"Copied annotations: {annotation_src} -> {annotation_dst}")
        else:
            print(f"Warning: Annotation file not found: {annotation_src}")
        
        # Copy images directory
        # Source: coco_dir/{split}2017/
        images_src = os.path.join(coco_dir, f'{coco_split}2017')
        
        if os.path.exists(images_src):
            # Copy all images from the source directory
            for img_file in os.listdir(images_src):
                img_src_path = os.path.join(images_src, img_file)
                img_dst_path = os.path.join(rf_split_dir, img_file)
                
                if os.path.isfile(img_src_path):
                    shutil.copy2(img_src_path, img_dst_path)
            
            print(f"Copied images from {images_src} to {rf_split_dir}")
        else:
            print(f"Warning: Images directory not found: {images_src}")
    
    print(f"RF-DETR dataset structure created at {rf_detr_dir}")
    
    

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
    
    print_dataset_info(dataset)
    dataset.splitter.StratifiedGroupShuffleSplit(train_pct=int(args.split[0])/100, val_pct=int(args.split[1])/100, test_pct=int(args.split[2])/100, batch_size=1)
    dataset.analyze.ShowClassSplits()
    
    yolo_lbl_dir = os.path.join(output_folder,'yolo','labels')
    dataset.export.ExportToYoloV5(output_path=yolo_lbl_dir,yaml_file='data.yaml', copy_images=True, use_splits=True,cat_id_index=0)
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
    
    #this function will just manipulate the already created yolo folder
    ultralytics_compatibility(os.path.join(output_folder,'yolo'),"data.yaml")
    
    rf_detr_dir = os.path.join(output_folder,'rf_detr')
    os.makedirs(rf_detr_dir,exist_ok=True)
    rf_detr_compatibility(coco_dir,rf_detr_dir)
    
    os.remove("tmp.json")