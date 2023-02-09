# Augmentation and Adjustment
## Improving Neural Image Captioning (from Show and Tell paper by Vinyals, et. al., 2014)

The goal of this project is to make use of new data augmentation and sampling methods to improve the Neural Image Captioning performance. By performing data augmentations, we can increase the size of the training data, which almost always helps with accuracy of neural network models. 

By leveraging Nucleus Sampling, we can attain more cohesive image captions that are actually descriptive, as compared to the BeamSearch approach taken in the original paper.

3 different models were trained:
- Our own implementation of the model with simple data augmentations
- Our own implementation with complex augmentations
- A model based on Shwetank Panwar's reimplementation done in 2015

Our architect differs in that it includes a deeper ResNet as the CNN encoder, layer normalization, and LSTM processing per word. We achieved an increase in BLEU-4 scores for captions generated for images in the test set as compared to the original NIC paper.
