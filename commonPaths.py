DATA_DIRECTORY = "/mnt/d/1. Studia/4. Semestr/Projekt Indywidualny/data/"
COCO_DIRECTORY = "cocoOriginalData/"
COCO_TRAIN_IMG_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "train/images/"
COCO_TRAIN_ANNOT_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "train/annotations/"
COCO_VAL_IMG_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "val/images/"
COCO_VAL_ANNOT_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "val/annotations/"
COCO_TEST_IMG_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "test/images/"
COCO_TEST_ANNOT_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "test/annotations/"
COCO_DUMP_VAL_IMG_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "dump/val/" 
COCO_DUMP_TRAIN_IMG_DIR = DATA_DIRECTORY + COCO_DIRECTORY + "dump/train/"

ANNOT_YOLO_TRAIN_JSON_FILE = COCO_TRAIN_ANNOT_DIR + "annotsForYoloTrain.json"
ANNOT_YOLO_VAL_JSON_FILE = COCO_VAL_ANNOT_DIR + "annotsForYoloVal.json"
VAL_ANNOT_FILENAME = "instances_val2017.json"
TRAIN_ANNOT_FILENAME = "instances_train2017.json"

MAP_RESULTS_VAL_DIR = DATA_DIRECTORY + "mAPResults/val/"
MAP_RESULTS_TRAIN_DIR = DATA_DIRECTORY + "mAPResults/train/"

SPEED_RESULTS_VAL_DIR = DATA_DIRECTORY + "speedTestResults/val/"
SPEED_RESULTS_TRAIN_DIR = DATA_DIRECTORY + "speedTestResults/train/"

SPEED_RESULTS_VAL_FILE = SPEED_RESULTS_VAL_DIR + "speedTestResults_val.csv"
SPEED_RESULTS_TRAIN_FILE = SPEED_RESULTS_TRAIN_DIR + "speedTestResults_train.csv"
PARAMETERS_DATA = DATA_DIRECTORY + "parametersCount/parametersCount.csv"

YOLO_DATA_DIR = DATA_DIRECTORY + "dataForYolo/"
YOLO_TRAIN_IMG_DIR = YOLO_DATA_DIR + "images/train/"
YOLO_VAL_IMG_DIR = YOLO_DATA_DIR + "images/val/"
YOLO_TEST_IMG_DIR = YOLO_DATA_DIR + "images/test/"
YOLO_TRAIN_LABELS_DIR = YOLO_DATA_DIR + "labels/train/"
YOLO_VAL_LABELS_DIR = YOLO_DATA_DIR + "labels/val/"
YOLO_TEST_LABELS_DIR = YOLO_DATA_DIR + "labels/test/"

YOLO_CONFIG_FILE = YOLO_DATA_DIR + "yoloConfig.yaml"