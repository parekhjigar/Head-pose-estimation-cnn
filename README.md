# Head Pose Estimation CNN

The Dataset used is modified version of the data available at: [Head Pose Image Database](http://www-prima.inrialpes.fr/perso/Gourier/Faces/HPDatabase.html)

The head pose database is a benchmark of 2790 monocular face images of 15 persons with variations of pan and tilt angles from -90 to +90 degrees. For every person, 2 series of 93 images (93 different poses) are available. The purpose of having 2 series per person is to be able to train and test algorithms on known and unknown faces (cf. sections 2 and 3). People in the database wear glasses or not and have various skin colour. Background is willingly neutral and uncluttered in order to focus on face operations.

The image dimensions are [192,144]

**Tilt (Vertical angle)** = {-90, -60, -30, -15, 0, +15, +30, +60, +90},  
Negative values - bottom, Positive values - top;


**Pan (Horizontal angle)** = {-90, -75, -60, -45, -30, -15, 0, +15, +30, +45, +60, +75, +90},  
Negative values - left, Positive values - right;
                .

## Files:

- `Dataset`: Contain all the images (test and train set).

- `train_data.csv`: Contain files names of the train set, person id, sequence id for each person, ground truth tilt and pan angles. This data is to be used in developing the models. 

- `test_data.csv`: Contain files names of the test set, person id, sequence id for each person. You need to predict the tilt/pan for this data and submit the prediction via canvas. The teaching team will use this data to evaluate the performance of the model you have developed.