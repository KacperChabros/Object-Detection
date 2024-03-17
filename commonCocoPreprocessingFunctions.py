import json

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
    categoryIdToName = {}
    for currentYoloId, category in enumerate(instancesJSON["categories"]):
        cocoId = category["id"]
        categoryName = category["name"]
        categoryIdToName[cocoId] = CocoCategory(cocoId, categoryName, currentYoloId)
    return categoryIdToName
