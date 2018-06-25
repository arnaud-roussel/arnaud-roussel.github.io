---
layout: post
title: Avito Kaggle competition - Demand Prediction - Part 1
---

In this two part blog post I go over my solution for the Avito challenge competition on Kaggle. It was a pretty interesting competition since it 
forced me to use many techniques across different fields in Machine Learning like Natural Language Processing and Computer vision. The solution is
divided into two parts:
* Part 1 focuses on explaining the problem and some of the feature engineering used
* Part 2 looks at the different models tried and also the stacking methodology

## The Problem: 

Avito is a russian online advertisement company. For people living in Canada, think Kijiji but russian. People want to sell items or services to others
and will therefore post online a description of the item, a picture and a price. The task is to predict whether or not an advertisement posted will lead
to a deal or not. It is important to note that Avito has a more complicated system than 1/0 for a deal, making the target variable a continuous number between
0 and 1 (leading to a regression task) rather than a binary variable. The training data given is 1.5 million ads and we have to predict 500K ads from
a test set. The quality of the predictions is evaluated with root mean squared error (RMSE) which is pretty common for regression tasks.

![_config.yml]({{ site.baseurl }}/images/Avito EDA 1_76_0.png)
*Some Images in the collection*

The data given to us is very rich:
* Some information about the advertisement like price, city, region, user (who posted).
* Title and Description texts.
* Image associated to the ad (maximum of 1 image per ad).
* A large dataset without the target (and which is not part of the test set either)

Note that the text is in russian, creating some additional difficulty for analysis. Let's look at an example :

![_config.yml]({{ site.baseurl }}/images/example_avito.PNG)
*An example in the dataset*

## Features

For exploratory data analysis I strongly suggest going over my notebook on github. See the markdown version [here](https://github.com/arroqc/Avito-Kaggle/blob/master/Avito%20EDA%201.md). It contains most of the important parts. 

**Missing Data**  
Early models show a big importance of some variables. Most notably price and image_top_1. Image top 1 is assumed to be some classification of the image for an advertisement. Since I
found a strong correlation with the target variable, it was deemed necessary to impute missing prices and image_top_1 with a model rather than a simple method like the mean. The way
it was modelled was to use a recurrent neural network trained on reading the texts and predict the image_top_1 category. Here is the architecture used:

![_config.yml]({{ site.baseurl }}/images/RNN_guess.PNG)
*RNN architecture for image_top_1 imputation*

A similar idea was used for price but simply adding the city as a potential factor for determining price. You can also find the notebooks for this imputation in the imputation subdirectory
of the [github repo](https://github.com/arroqc/Avito-Kaggle) As you have noticed, an embedding layer for words is used. The embedding have been trained on the larger dataset that doesn't contain
the target variable. It is rich in descriptions so a simply word 2 vec algorithm was made on its corpus of text using gensim library. To know more about word2vec and the different 
ways of finding word embeddings you can view [this page for example](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/).

**Text Features**  
Two categories of features are derived from the text data. The meta text features and features based on content. The meta text features contain things like:
* Length of the title or description
* Number of uppercase (maybe too much uppercase is detrimental)
* Ratio of length between title and description
* Ratio of unique words
* etc.

The gain in predictive power out of these features is small but it's a gain nonetheless. For the text itself, when used in a neural network it is used as is (after tokenization) but
for classical machine learning like trees we need to transform the text. A simple TF-IDF method is used for the description. Term Frequency inverse document frequency (TF-IDF) simply
 counts the number of occurences of each word in each texts and divides by the number of time the word appears at least one across texts. Here is an illustration:
 
![_config.yml]({{ site.baseurl }}/images/tfidf.PNG)
*TF-IDF illustration*

As you can see it creates a very long vector of numbers for each document. That vector is used as a feature in another model. It may sound like too many features (sometimes in the millions) 
but the vector is usually sparse (full of 0) so some algorithms are able to use that fact to have efficient ways of dealing with this long vector.

**Image Features**  
Due to a lack of time images were dealt with in two simple ways. As for text, we have meta image features:
* Height, Width
* Size
* Blurness
* Brightness
* Dominant colors

These ended up helping very little. In a neural network one way to deal with it would be to build a convolutional neural net on images before concatenating with other features but this 
requires a lot of computing power and I lacked time. Instead a simple approach was used: using pretrained models. The Keras library offers pretrained models that perform well on
an image classification task called ImageNet. The idea was to simply load them and make them predict the image class of the images. To reduce variance 3 models are used (Resnet, Inception, Exception)
and the top 3 classes are considered.

![_config.yml]({{ site.baseurl }}/images/inception.png)
*Inception model [Source Google AI blog](https://ai.googleblog.com/2016/03/train-your-own-image-classifier-with.html)*

Model used in part 2

