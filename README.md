# StackOverflow Tag Prediction:

### Summary:
Document classification is used to divide and organize text for fast, efficient retrieval in the future. This project explores various NLP techniques to automate this classification task on a dataset from Stack Overflow.
Stack Overflow is a knowledge sharing platform where users may post questions and answers. Each question is manually tagged with categories by the user, known as “tags”. I will build models that can correctly predict the appropriate tag(s) for a given question based solely on the text in the title and body of a given post.


### Process:

The dataset was obtained from Kaggle. This analysis will look at the top 15 most frequently used tags in the dataset (~1M questions total).

#### Pipeline is as follows:
1. Process/Clean Text Data
2. Vectorize Text Data
3. Build Models - Predict on Vectorized Text

### Results:
As we are dealing with a multi-label classification problem, the F1 (macro) score was used as a metric in evaluating model efficiency.

4 traditional algorithms were considered with a TFIDF vectorizer. Multinomial Naive Bayes, Random Forest, Logistic Regression and Linear SVC. I also used a 2 layer 1 dimensional CNN with a fully connected ANN output layer in conjunction with a Word2Vec Skipgram model. The CNN model outperformed all other models with a F1 score of 0.77.
