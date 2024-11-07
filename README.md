# DrIDENT: THE APP FOR PILL RECOGNITION
A model designed to recognize solid pharmaceuticals 

### THE DATASETS

Dataset 1:
- https://datadiscovery.nlm.nih.gov/Drugs-and-Chemicals/Computational-Photography-Project-for-Pill-Identif/5jdf-gdqh/about_data (provided by the National Library of Medicine)
Dataset 2:
- https://www.kaggle.com/datasets/trumedicines/pharmaceutical-tablets-dataset (provided by Gaurav Dutta on Kaggle)

NOTE:
Due to the size of the dataset the datasets must be downloaded if the code should be run.

How to run the code:
- Repository File Overview:
  - DrIDENT-code.py
    > needs the following files to run:
      - metadata: table.csv from Dataset 1, Training_set.csv from Dataset 2
      - images in folders: original from Dataset 1, train from Dataset 2
      After downloading the datasets please insert the path to both the metadata and the images in order to run the code.
    > Please make sure the following packages are installed:
      - numpy
      - pandas
      - cv2
      - os
      - math
      - matplotlib.pyplot
      - tensorflow
      - pickle
      - scikit-learn
      - flask
      - tf2onnx
      - PIL
      - io
  - DrIDENT.pptx
  - DrIDENT.pdf
