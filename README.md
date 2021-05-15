# Hands-On-Machine-Learning
<h2> 1. Classification </h2> 
<br>  It is the process of identifying to which set of categories a new observation belongs.Here On classification we used the library Sckit learn.
Datasets loaded by Scikit-Learn generally have a similar dictionary structure including: 

<li>  A DESCR key describing the dataset </li>
<li>  A data key containing an array with one row per instance and one column per feature </li>
<li>  A target key containing an array with the labels </li>
<br>
Looking at the arrays we got to know there are 70,000 images, and each image has 784 features. This is because each image
is 28×28 pixels, and each feature simply represents one pixel’s intensity, from 0
(white) to 255 (black). Let’s take a peek at one digit from the dataset. All you need to
do is grab an instance’s feature vector, reshape it to a 28×28 array, and display it using
Matplotlib’s imshow() function.


[!Image]<img scr=https://res.cloudinary.com/adeshpokhrel/image/upload/v1621076062/1_v7nx5p.png></img>
