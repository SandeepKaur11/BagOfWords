
# IMAGE CLASSIFICATION USING BAG OF WORDS

The bag-of-words model is a way of representing text data when modeling text with machine learning algorithms.The bag-of-words model is simple to understand and implement and has seen great success in problems such as language modeling and document classification.
In feature quantization, we executed the commands matlab file and obtained the two variables, TrainMat(270000*128) and TestMat(90000*128) are loaded. The sift features has 128
dimensions. After Dictionary creation of 500 words which is controlled by Dictionary Size variable paramter, we obtained values in matlab file. We obtained Euclidean distance and assign a descriptor in the training and test images to the nearest codeword cluster. 
![alt text](Demo/BoWIageTesting.PNG?raw=true)
## Image Classification using Nearest Neighbour(NN) classifier
![alt text](Demo/BoWTestNNImage.PNG?raw=true)
We can have visual words representation of each image in the training and testing dataset as a histogram.
![alt text](Demo/CodewordHistogram.PNG?raw=true)

![alt text](Demo/CodewordHistogram1.PNG?raw=true)

![alt text](Demo/ConfusionMatrix.PNG?raw=true)

![alt text](Demo/FD.PNG?raw=true)













