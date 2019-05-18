# SMT-NET
Reduction of training a neural network into an SMT problem

Task description. In the Optical Character Recognition (OCR) we seek to predict a number (between 0 to 9) for a given image of handwritten digit. In this assignment we simplify the OCR into a binary classification task. Specifically, we consider predictions for only two numbers 3 and 5. 

# Data
All the handwritten digits are originally taken from http://www.kaggle.com/c/digit-recognizer/data The original dataset contains the sample digits suitable for OCR. We extract samples with only labels 3 and 5. Following a little pre-processings we produce three datasets for this assignment as follows:
# (a) 
Train Set (pa2_train.csv): Includes 4888 rows (samples). Each sample is in fact a list of 785 values. The first number is the digit's label which is 3 or 5. The other 784 floating values are the the attened gray-scale values from a 2d digital handwritten image with shape 28 x 28.
# (b) 
Validation Set (pa2_valid.csv): Includes 1629 rows. Each row obeys the same format given for the train set. This set will be used to select your best trained model.
# (c) 
Test Set (pa2_test.csv): Includes 1629 rows. Each row contains only 784 numbers. The label column is omitted from each row.
