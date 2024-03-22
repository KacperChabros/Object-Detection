import torch
from collections import Counter
from collections import defaultdict

def intersectionOverUnion(bboxPred, gtBbox):
    '''
        ### intersectionOverUnion
        function that calculates intersection over union for two given bounding boxes

        :param bboxPred: [xMid, yMid, width, height] (torch.tensor)
        :param gtBbox: [xMid, yMid, width, height] (torch.tensor)
        :return: numerical value of IoU between two boxes. (torch size of 1)
    '''
    detBboxX1 = bboxPred[0] - bboxPred[2] / 2
    detBboxY1 = bboxPred[1] - bboxPred[3] / 2
    detBboxX2 = bboxPred[0] + bboxPred[2] / 2
    detBboxY2 = bboxPred[1] + bboxPred[3] / 2

    gtBboxX1 = gtBbox[0] - gtBbox[2] / 2
    gtBboxY1 = gtBbox[1] - gtBbox[3] / 2
    gtBboxX2 = gtBbox[0] + gtBbox[2] / 2
    gtBboxY2 = gtBbox[1] + gtBbox[3] / 2

    x1 = torch.max(detBboxX1, gtBboxX1)
    y1 = torch.max(detBboxY1, gtBboxY1)
    x2 = torch.min(detBboxX2, gtBboxX2)
    y2 = torch.min(detBboxY2, gtBboxY2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    detBboxArea = abs((detBboxX2 - detBboxX1) * (detBboxY2 - detBboxY1))
    gtBboxArea = abs((gtBboxX2 - gtBboxX1) * (gtBboxY2 - gtBboxY1))

    return intersection / (detBboxArea + gtBboxArea - intersection)


def meanAveragePrecission(predBboxes, gtBboxes, iouThreshold=0.5, numClasses=80):
    '''
        ### meanAveragePrecission
        calculates the mean average precision for given pred and gt bboxes and given threshold

        :param predBboxes: list of lists in the following format: [imageId, classPred, xMid, yMid, width, height, probScore]
        :param gtBboxes: list of lists in the following format: [imageId, class, xMid, yMid, width, height]
        :param iouThreshold: IoU threshold to decide whether two boxes represent the same object. By default set to 0.5
        :param numClasses: number of classes in the dataset. By default set to 80 (Coco dataset)
        :return: value of mAP for the given set of pred and gt bboxes and given threshold (torch.tensor size of 1)
    '''
    averagePrecissionForClass = []

    # to group predictions by classId (same for ground truths)
    categoryIdToPredBboxes = defaultdict(list)
    categoryIdToGTBboxes = defaultdict(list)

    for detection in predBboxes:
        categoryId = detection[1]
        categoryIdToPredBboxes[categoryId].append(detection)

    for groundTruth in gtBboxes:
        categoryId = groundTruth[1]
        categoryIdToGTBboxes[categoryId].append(groundTruth)

    #calculate for each class
    for cat in range(numClasses):
        detectionsForClass = categoryIdToPredBboxes[cat]
        groundTruthsForClass = categoryIdToGTBboxes[cat]

        #the line below will produce a dictionary with Key=ImageId and Value=how many true bounding boxes it has
        ammountOfBboxes = Counter([gt[0] for gt in groundTruthsForClass])

        # this will create a tensor full of zeros instead of just a number of bounding boxes
        # we do this to keep track of the true bboxes that we have covered so far (only the first pred bbox
        # that covers the gt bbox is TP, others are FP) 0 = not covered, 1 = covered
        for imageId, noBboxes in ammountOfBboxes.items():
            ammountOfBboxes[imageId] = torch.zeros(noBboxes)

        detectionsForClass.sort(key=lambda x: x[6], reverse=True) # sorting predictions by the desc confidence

        # to keep track of which detection is TP:   0-it is not, 1-it is TP
        #(each class has its own TPs with length of the number of predicted bboxes for this class) 
        # 1 row - number of detection bboxes columns
        TPs = torch.zeros(len(detectionsForClass))
        FPs = torch.zeros(len(detectionsForClass))

        amountOfTrueBboxes = len(groundTruthsForClass)
        # amountOfTrueBboxes = len(gtBboxes)  #number of ground truth bboxes across all images

        if(amountOfTrueBboxes == 0):
            if(len(detectionsForClass) == 0):
                averagePrecissionForClass.append(torch.tensor(1))
            else:
                averagePrecissionForClass.append(torch.tensor(0))
            continue
        for detectionId, detection in enumerate(detectionsForClass): #here we've got particular bbox for particular class
            # we take only those gt bboxes which belong to the same img as considered detection bbox
            gtsForSameImg = [
                bbox for bbox in groundTruthsForClass if bbox[0] == detection[0]
            ]
            # noTrueBboxesInThisImg = len(gtsForSameImg) # number of true bboxes in the considered img

            maxIoU = 0
            bestGtId = None

            for gtId, gt in enumerate(gtsForSameImg):#compare taken detection with all GTbboxes from the same img
                iou = intersectionOverUnion(    #calculate IoU for them
                    torch.tensor(detection[2:6]),
                    torch.tensor(gt[2:])
                )

                if (iou > maxIoU): #check if its better than current max
                    maxIoU = iou
                    bestGtId = gtId

            #after checking with all gt bboxes for this img check if maxIoU is greater than threshold
            if (maxIoU > iouThreshold): 
                #check if the potential matching GT bbox hasn't been assigned to another detection bbox yet
                if ( ammountOfBboxes[detection[0]][bestGtId] == 0 ): #(bbox number bestGtId in img detection[0])
                    TPs[detectionId] = 1
                    ammountOfBboxes[detection[0]][bestGtId] = 1
                else:
                    FPs[detectionId] = 1
            else:
                FPs[detectionId] = 1

        #calculate cumsum of TPs and FPs
        TPsCumsum = torch.cumsum(TPs, dim=0)
        FPsCumsum = torch.cumsum(FPs, dim=0)
        

        recalls = (TPsCumsum / amountOfTrueBboxes)
        precisions = TPsCumsum / (TPsCumsum + FPsCumsum)

        #we add point (0,1) in order to provide smooth start to the chart
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))

        # calculate area under the precision-recall curve
        averagePrecissionForClass.append((torch.trapz(precisions, recalls)).clamp(0))
    return averagePrecissionForClass
    # return float(sum(averagePrecissionForClass) / len(averagePrecissionForClass))