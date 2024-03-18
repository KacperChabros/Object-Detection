import json
import shutil
import os
from collections import defaultdict
from collections import OrderedDict

class CocoImage:
    def __init__(self, id, fileName, height, width):
        self.id = id
        self.fileName = fileName
        self.height = height
        self.width = width
        self.annotations = list()


class CocoCategory:
    def __init__(self, cocoId, categoryName, yoloId):
        self.cocoId = cocoId
        self.categoryName = categoryName
        self.yoloId = yoloId

def getInstancesAsJSON(pathToAnnotationsDir, fileName):
    '''
        ### getInstancesAsJSON
        opens the given annotation file and parses it as JSON
        :param pathToAnnotationsDir: path to instances folder (with trailing slash)
        :param fileName: name of the file containing coco instances
        :return: parsed instances file as JSON
    '''
    pathToFile = pathToAnnotationsDir+fileName
    instancesFile= open(pathToFile)
    instancesJSON = json.load(instancesFile)
    return instancesJSON


def associateImageIdWithItsProperties(instancesJSON):
    '''
        ### associateImageIdWithItsProperties
        creates a dictionary which contains ImageId (key) and a CocoImage object with its properties (value).
        This function doesn't fill the object with the annotations, it only creates an empty list.
        To fill the list, use fillImagesWithAnnotations()

        :param instancesJSON: file with annotations parsed as JSON (e.g. result of getAnnotationsAsJSON)
        :return: dictionary: Key=ImageId    Value=Initialized CocoImage object (without the annotations)
    '''
    imageIdToProps = {}
    for image in instancesJSON["images"]:
        id = image["id"]
        fileName = image["file_name"]
        width = image["width"]
        height = image["height"]
        imageIdToProps[id] = CocoImage(id, fileName, height, width)
    return imageIdToProps

def fillImagesWithAnnotations(instancesJSON, imageIdToProps):
    '''
        ### fillImagesWithAnnotations
        fills objects (annotations list property) in the given dictionary with corresponding annotations

        :param instancesJSON: file with annotations parsed as JSON (e.g. result of getAnnotationsAsJSON)
        :param imageIdToProps: dictionary which contains ImageId (key) and a CocoImage object with its properties (value). 
        (e.g.) result of the associateImageIdWithItsProperties function
        :return: dictionary: Key=ImageId    Value=CocoImage object with filled annotations property
    '''
    for ann in instancesJSON["annotations"]:
        imageId = ann["image_id"]
        imageIdToProps[imageId].annotations.append(ann)
    return imageIdToProps


def associateImageIdWithItsPropsAndAnnots(instancesJSON):
    '''
        ### associateImageIdWithItsPropsAndAnnots
        creates a dictionary that associates ImageId with its properties (id, fileName, width, height,
        corresponding annotations)

        :param instancesJSON: file with annotations parsed as JSON (e.g. result of getAnnotationsAsJSON)
        :return: dictionary: Key=ImageId    Value=CocoImage object
    '''
    imageIdToPropsAndAnnots = associateImageIdWithItsProperties(instancesJSON)
    imageIdToPropsAndAnnots = fillImagesWithAnnotations(instancesJSON, imageIdToPropsAndAnnots)
    return imageIdToPropsAndAnnots


def associateCategoryIdWithItsNameAndYoloId(instancesJSON):
    '''
        ### associateCategoryIdWithItsNameAndYoloId
        creates a dictionary which contains CategoryId (key) and its name (value)

        :param instancesJSON: file with annotations parsed as JSON (e.g. result of getAnnotationsAsJSON)
        :return: dictionary: Key=CategoryId    Value= CocoCategory object containing cocoId, categoryName and yoloId
    '''
    categoryIdToNameAndYoloId = {}
    for currentYoloId, category in enumerate(instancesJSON["categories"]):
        cocoId = category["id"]
        categoryName = category["name"]
        categoryIdToNameAndYoloId[cocoId] = CocoCategory(cocoId, categoryName, currentYoloId)
    return categoryIdToNameAndYoloId



def clearDataSetFromNotAnnotatedImgs(sourceDir, destDir, imageIdToPropsAndAnnots, safe=True):
    '''
        ### clearDataSetFromNotAnnotatedImgs
        clears the desired directory from the images that were not annotated. In Coco dataset they are mostly
        road signs. The function also clears imageIdToPropsAndAnnots from not annotated images
        :param sourceDir: directory with the images (with trailing slash)
        :param destDir: directory for where to move not annotated files to (with trailing slash)
        :param imageIdToPropsAndAnnots: dictionary: Key=ImageId    Value=CocoImage object (cleared of not annotated images)
        (e.g.) result of associateCategoryIdWithItsNameAndYoloId
        :param safe: whether to stop executing when the destDir contains files (set to True by default) (if False,
        some files may be overwritten)
        :return: dictionary: Key=ImageId    Value=CocoImage object (cleared of not annotated images)
    '''
    if(safe):
        numberOfFilesInDir = len(os.listdir(destDir))
        if( numberOfFilesInDir != 0 ):
            msg = f"The destination directory \"{destDir}\" is not empty! If you wish to move not annotated files anyway"
            msg += " run the function with safe=False. Some files may be overwritten!"
            # return imageIdToPropsAndAnnots
            raise Exception(msg)
    keysToDelete = []
    for imageId, image in imageIdToPropsAndAnnots.items():
        fileName = image.fileName
        numberOfAnnotations = len(image.annotations)
        if numberOfAnnotations == 0:
            shutil.move(sourceDir+fileName, destDir+fileName)
            keysToDelete.append(imageId)
    
    
    for key in keysToDelete:
        imageIdToPropsAndAnnots.pop(key)
    return imageIdToPropsAndAnnots

def createAnnotJSONForYolo(categoryIdToNameAndYoloId, imageIdToPropsAndAnnots, pathToJSONoutfile, safe=True):
    '''
        ### createAnnotJSONForYolo
        creates JSON file with annotations corresponding to the YOLO model output. It is done to 
        help with calculating mAP later. the format is:
        {"imageId": [ {"yoloCatId": ycId, "bbox": [xMid, yMid, width, height]}, {(...)anotherAnnot(...)} ]}

        :param categoryIdToNameAndYoloId: Key=CategoryId    Value= CocoCategory object containing cocoId, categoryName and yoloId
        (e.g.) result of associateCategoryIdWithItsNameAndYoloId function
        :param imageIdToPropsAndAnnots: dictionary: Key=ImageId    Value=CocoImage object (cleared of not annotated images)
        (e.g.) result of associateImageIdWithItsPropsAndAnnots function
        :param pathToJSONoutfile: path where to place JSON file
        :param safe: whether to check if the JSON file exists (when safe=False, the file may be overwritten!)
    '''
    if(safe):
        if(os.path.exists(pathToJSONoutfile)):
            msg = f"\"{pathToJSONoutfile}\" already exists! If you want to overwrite it, run the function with safe=False"
            raise Exception(msg)

    sortedImageIdToAnnots = OrderedDict(sorted(imageIdToPropsAndAnnots.items()))
    imageIdToListOfAnnotsSerializable = defaultdict(list)
    for imageId,image in sortedImageIdToAnnots.items():
        imgWidth = image.width
        imgHeight = image.height
        for ann in image.annotations:
            annotDict = {}
            cocoCatId = ann["category_id"]
            yoloCatId = categoryIdToNameAndYoloId[cocoCatId].yoloId
            bbox = ann["bbox"]
            x = float(bbox[0])
            y = float(bbox[1])
            bboxWidth = float(bbox[2])
            bboxHeight = float(bbox[3])
            x = round((x + bboxWidth / 2) / imgWidth,6)
            y = round((y + bboxHeight / 2) / imgHeight,6)
            bboxWidth = round(bboxWidth / imgWidth,6)
            bboxHeight = round(bboxHeight / imgHeight,6)
            yoloBbox = [x, y, bboxWidth, bboxHeight]

            annotDict["yoloCatId"] = yoloCatId
            annotDict["bbox"] = yoloBbox
            imageIdToListOfAnnotsSerializable[imageId].append(annotDict)


    with open(pathToJSONoutfile, "w") as outfile:
        json.dump(imageIdToListOfAnnotsSerializable, outfile)