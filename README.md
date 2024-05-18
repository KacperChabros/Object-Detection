## Collection and analysis of performance and accuracy of YOLOv8 models
This project was created as my Individual Project at the University. The goal was to learn how the Convolutional Neural Networks work and analyse how different YOLOv8 models perform.  
All of the tests were performed on MS COCO dataset for Object Detection.  
The project focuses on preprocessing the images from the COCO format to the format accepted by YOLO models. Moreover, it performs speed tests as well as it calculates mean average precision for the pretrained models based on their predictions. The obtained data is analysed as the last part of this project.  
### Technologies
* Python and its libraries:
    * numpy
    * pandas
    * torch
    * matplotlib
    * ultralytics (YOLO models)
### Features
* Displaying images with bounding boxes
* Preprocessing of MS COCO data to YOLO format
* Calculating mean average precision on the predictions made by pretrained models
* Performing speed tests on models
* Saving results to .csv files
* Analysis of the results
* Generating plots useful for understanding the collected data.

### Setup and usage
After installing the necessary libraries one can simply run/browse the notebooks. Below is a short description which notebook serves what purpose:  
* **_CocoImagePresenter_** - the notebook to get familiar with the COCO dataset. Displays images with the bounding boxes
* **_CocoDatasetPreprocessor_** - preprocesses data from COCO to YOLO format.
* **_YoloMAPTests_** - performs predictions on the COCO dataset. The predictions are used to calculate mAP against the ground truths.
* **_YoloSpeedTests_** - performs speed tests on the COCO dataset. Measures time needed by the models to complete the predictions.
* **_ResultsAnalysis_** - gathers the data collected in previous notebooks and analyses it with the help of exploratory analysis.

### Useful links
* [MS COCO dataset](https://cocodataset.org/#download)
* [YOLOv8 GitHub project](https://github.com/ultralytics/ultralytics)