# Malaria-CELL-Detection-KERAS-CNN
The data can be downloaded from the following Kaggle dataset  https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria .  
Link to my Kaggle kernel https://www.kaggle.com/parthsharma5795/malaria-detection-keras-cnn-95-accuracy

Code Requirements

Numpy
Pandas
cv2
Seaborn,matplotlib
Keras


Description

This is an image classification problem on Kaggle Datasets.The dataset contains 2 folders - Infected - Uninfected and has been originally taken from a government data website  https://ceb.nlm.nih.gov/repositories/malaria-datasets/ .



Breakdown of the code:

1.Loading the dataset: Load the data and import the libraries.
2.Data Preprocessing:
   Reading the images,labels stored in 2 folders(Parasitized,Uninfected).
   Plotting the Uninfected and Parasitized images with their respective labels.
   Normalizing the image data.
   Train,test split
3.Data Augmentation: Augment the train and validation data using ImageDataGenerator
4.Creating and Training the Model: Create a cnn model in KERAS.
5.Evaluation: Display the plots from the training history.
6.Submission: Run predictions with model.predict, and create confusion matrix.



Results:

The accuracy for the test dataset came out to be 97%. The link to my kaggle kernel is https://www.kaggle.com/parthsharma5795/malaria-detection-keras-cnn-95-accuracy
