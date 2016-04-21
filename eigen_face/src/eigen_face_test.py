"""
Image classification using Eigen face.
Dataset: The Extended Yale Face Database B 
Folder structure: 
	src/this_file.py
	CroppedYale/
"""
import cv2
import glob
import numpy as np
import random
from sklearn.decomposition import RandomizedPCA
import math

"""
Images are of different resolution. 
Taking only the ones with this resolution.
"""
IMG_RES = (192 , 168)

"""
Run with different values of Eigen faces.
"""
#NUM_EIGENFACES = 200 #1 case
#NUM_EIGENFACES = len(glob.glob('../CroppedYale/yale*')) #2nd case
NUM_EIGENFACES = len(glob.glob('../CroppedYale/yale*'))*10 #3rd case

"""
Function to create matrix for PCA.
Input: glob_path: List of images
       image_size: the standard size of the image
Output: Image matrix, corresponding label
"""
def get_data_from_images(glob_path, image_size):
  arr1 = None
  training_labels = []
  
  for filename in glob_path: 
    arr = cv2.imread(filename, 0)
    if arr.shape == image_size:
      training_labels.append(filename.split('/')[3].split('_')[0])
      curr_shape = arr.shape
      arr = arr.reshape(1, curr_shape[0]*curr_shape[1])
      if arr1 is not None:
        arr1 = np.append(arr1,arr,axis=0)
	continue
      arr1 = np.array(arr)
  return arr1, training_labels

"""
Function to print confusion matrix.
Input: test_result: List of result given by method/classifier
       correct_result: Expected/Target result List
       print_mat: boolean for printing the confusion matrix.
Output: Accuracy	  
"""
def print_confusion_matrix(test_result, correct_result, print_mat = False):

    conf_dict = {}
    correct_count = 0

    objects = list(set(correct_result))

    for i in objects:
        conf_dict[i]={}
        for j in objects:
            conf_dict[i][j]=0

    for i in range(len(test_result)):
        conf_dict[correct_result[i]][test_result[i]]+=1

    for i in range(len(test_result)):
        if test_result[i] == correct_result[i]:
            correct_count += 1;

    accuracy = float(correct_count)*100/len(test_result)

    print correct_count," classified correctly out of ",len(test_result)
    print "Accuracy: ", accuracy,"%"

    if print_mat:
	print "Confusion Matrix:"
        print "Actual | Predicted:"
        for i in objects:
            print i,":",conf_dict[i]
    return accuracy

"""
This is main.
"""
if __name__ == "__main__":

  usable_image = []
  FOLDS = 5

  """
  test and training datasets have been created here.
  """
  folders  = glob.glob("../CroppedYale/yale*")
  for folder in folders:
    images = glob.glob(folder+"/*.pgm")
    for image in images:
      arr = cv2.imread(image, 0)
      if arr.shape == IMG_RES:
	usable_image.append(image)

  total_len = len(usable_image)

  for i in range(FOLDS):
    print "Fold ___",i+1
    random.shuffle(usable_image)
    testing_set = usable_image[:total_len/FOLDS]
    training_set = usable_image[total_len/FOLDS:]

    X_train, Y_train = get_data_from_images(training_set, IMG_RES)
    pca = RandomizedPCA(n_components=NUM_EIGENFACES, whiten=True).fit(X_train)
    X_pca = pca.transform(X_train)

    X_test, Y_test = get_data_from_images(testing_set, IMG_RES)
    predicted_labels = []

    for j, ref_pca in enumerate(pca.transform(X_test)):
      distances = []
    
      for i, test_pca in enumerate(X_pca):
          dist = math.sqrt(sum([diff**2 for diff in (ref_pca - test_pca)]))
          distances.append((dist, Y_train[i]))
 
      pred_label = min(distances)[1]
      predicted_labels.append(str(pred_label))

    print_confusion_matrix(predicted_labels, Y_test)

