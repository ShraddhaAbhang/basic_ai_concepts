'''
Deep Dive into Machine Learning and Deep Learning: Key Terms and Real-World Applications

Explore the fundamentals and advanced concepts of machine learning and deep learning in this 
comprehensive guide. From algorithms to evaluation metrics, we cover it all, providing clear explanations 
and real-world examples to help you understand these complex topics. Whether you’re new to the field or looking 
to deepen your expertise, this blog has something for everyone interested in AI and its applications.


1. PCA (Principal Component Analysis): 
→ PCA is a dimensionality reduction technique, PCA transforms the data into a set of orthogonal components that 
capture the most variance in the data, making it easier to visualize and process.if we have many similar features 
pca can be used to to reduce dimensionality and have fewer features which represents the all those features.

2. Feature Selection: →
→ feature selection is the process of selecting a subset of relevant features for use in a model, and getting rid 
of noise in data using Correlation matrix technique.

3. BERT: →
→ BERT (Bidirectional Encoder Representations from Transformers) is a transformer-based model designed for natural 
language processing (NLP) tasks. BERT is pre-trained on a large corpus of text in a bidirectional manner, meaning 
it considers the context from both left and right directions. This allows it to understand the nuanced meaning of 
words in context. BERT is used in search engines to better understand user queries and provide more relevant results.

4. Vanishing Gradient: →
→ In vanishing gradients the model’s weights are updated very slowly, making it difficult for the network to learn 
results. 

5. Exploding Gradient: →
→ When weights change fast & frequently the exploding gradient occurs.  These can cause the model’s weights to grow 
uncontrollably, leading to instability and poor model performance. Techniques like gradient clipping are used to 
address this issue.

6. Gradient Boosting: →
→ Gradient boosting improves predictions step by step. Each new model learns from the mistakes of the previous one, 
making the overall prediction more accurate. This helps the model learn complex patterns and make better predictions.

7. Random Forest: →
→ Random forest improves predictions by using many decision trees instead of just one. Each tree learns from a random 
part of the data, and the final result is based on the average (for numbers) or majority vote (for categories). 
This makes the model more accurate and reduces overfitting.
Random forests are used in credit scoring to predict the likelihood of default. 
By aggregating the predictions of multiple decision trees, the model provides more robust and accurate predictions.

8. Logistic Regression: →
→ Logistic regression is a statistical model used for binary classification tasks.  Logistic regression is a method used
 to classify things into two groups, like yes/no or true/false.It predicts the probability of something happening and 
 decides based on a set threshold.
For example, doctors use it to check if a patient has diabetes based on factors like age, BMI, and blood pressure.

9. Linear Regression: →
→ Linear regression is a statistical method for modeling the relationship between a dependent variable and one or more 
independent variables. used for predicting continuous outcomes. The model assumes a linear relationship and fits a linear 
line. 

10. LSTM (Long Short-Term Memory): →
→ LSTM is a special type of recurrent neural network that helps understand patterns in sequence data over time.It solves 
the vanishing gradient problem of forgetting old information by using memory cells that store important details for longer.
For example, LSTMs are used to predict stock prices and for language tasks like translation and text generation.

11. LLM (Large Language Model): →
→ LLMs (Large Language Models) are advanced NLP models trained on vast amounts of text to understand and generate human 
language.They use deep learning techniques, like transformers, to recognize patterns, context, and meaning, enabling them to 
perform various language tasks with minimal fine-tuning.
For example, models like GPT-3 can generate text, power chatbots, assist in automated content creation, and even write code.

12. Variance: →
→ Variance measures the spread of data points around the mean in a dataset. High variance indicates that data points are 
spread out widely, while low variance indicates that data points are close to the mean. It’s important for understanding 
data distribution and variability.

13. YOLO (You Only Look Once): →
→ YOLO is a real-time object detection algorithm that detects objects in images with high speed and accuracy based on 
convolutional neural networks. YOLO splits the image into a grid and, in one go, predicts bounding boxes and class 
probabilities for each grid cell. Non-maximum suppression (NMS) is a post-processing technique that is used in object 
detection tasks to eliminate duplicate detections and select bounding boxes. We remove the box when it has less than 
0.5 overlapping % using the IOU formula.

14. CNN (Convolutional Neural Network): →
→ CNNs is a class of deep neural networks designed to process structured grid data like images. CNNs use filters to learn 
important patterns from images, making them great for tasks like image classification and recognition.


15. RCNN (Region-based CNN): →
→ RCNN is an object detection method (algorithm) that finds and identifies objects in images using CNNs. RCNN first suggests 
possible object regions, then uses a CNN to classify and refine them, improving accuracy.
For example, RCNN is used in surveillance to detect cars and people in videos, helping with automated monitoring and alerts.

16. Transformers: →
→ Transformers are a type of neural network architecture designed for handling sequential data. They use self-attention to find 
relationships between words, work in parallel, and capture long-range dependencies. Transformers have revolutionized NLP by improving
performance on tasks like translation and text generation. BERT and GPT are based on transformers and are used for a wide range of 
NLP tasks, including question answering, language translation, and text summarization.

'''