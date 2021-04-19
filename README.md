# Introduction and problem statement

There are four total notebooks for this project. Over these four notebooks we gather data (Notebook 1), clean and analyze data (Notebook 2), develop models (Notebook 3) and develop a Streamlit app (Notebook 4). The work here is done in order to address the problem statement below. In this notebook we develop a function to pull posts from the pushshift.io Reddit api in the subreddit categoires r/OCD and r/Anxiety. We merge and export the dataframe for use in future notebooks.

## Problem statement

Can we leverage anonymized, free-text posts made in an open, safe environment from individuals discussing obsessive compulsive or anxiety disorders in order to build a first-step (non-medical) diagnostic tool?

# Cleaning and EDA
In Notebook 2 we first clean the data, dealing with null values and replacing where necessary. In step 2 we do extensive EDA including word count analysis, analysis of most common words, sentiment analysis and examination of interesting text features (eg. URLs).

During the EDA process we create a bespoke list of stopwords as, given the colloqiual nature of a lot of the posts, many of the most frequent words were words that likely have little predictive value but were not classified as english stopwords (eg. 'just' and 'like' were top 5 words in both subreddits!) In Notebook 3 these will be fed into the grid search as parameters.

We also do sentiment Analysis in order to understand the overall sentiment in the posts and compare that with subreddit specific sentiment. This allows us to get a sense of average compound sentiment (mean) and assess composition of the posts as well as identify sources of potential bias toward negative words in the data. This is relevant to the purpose of this analysis as we will be asking people to describe what it feels like in their mind when they are off-balance which is almost definitely going to carry a negative compound sentiment. So for purposes of prediction, if the sentiment in these posts was overwhelmingly positive, we might have a bad fit between the data we're training/testing on and the actual (non-medical) diagnostic tool.

# Modeling

In Notebook 3 we develop three models with both CountVectorization and Tfidf Vectorization (six total models) in order to predict whether combined title + selftext posts come from OCD or Anxiety subreddits. Throughout the model pipelines we offer the gridsearch the opportunity to select words from our own stop words list that was developed in the previous notebook.

After model development we take the best model (the logistic regression with tfidf vectorization) and look at posts that we misclassified. We create a separate stopwords classification with all of these words and feed them back into the parameters. When given the choice the model does not select from this stopword list. When not given the choice (between none, english and my_stopwords3) and only my_stopwords3 the accuracty decreases. Next, we pickle the log reg tfidf pipeline and pass it onto Notebook 4 to create a streamlit app.

# Streamlit app

In Notebook 4 we build a streamlit app that could be used as a first step non-medical diagnostic tool. The model that we bring in here from a previously pickled item is the log reg with tfidf production model explained in Notebook 3.

The streamlit app can be seen in this project folder under the name app_v2.py 

# Conclusions and Further Steps
Overall, the model selected for production was the log reg with tfidf which had a training score of .956 and testing score of .908. After extensive EDA in Notebook 2 we can make several interesting observations about the data, perhaps most important to the project statement: that the sentiment of each subreddit and the dataframe as a whole was negative.

This sets the scene for the Streamlit app based off a pickle from our production model that poses the following question to the user: **'Tell us about your internal mental environment on days/at times when you feel off balance.'** In the Streamlit app we also included a sentiment analysis given that, if this were to be used as pre-assessment in a clinical environment, it may be useful to the caregiver to have information about the overall sentiment of the user's response to the question that led to the non-medical, first-stage diagnosis.

In response to the initial problem statement of whether or not we can develop a non-medical, first-step diagnostic tool, the answer is **yes** and we have reasonable grounds to believe it may be fairly accurate for some users. Again, this is non-medical and should not replace guidance from a trained medical professional but it is, perhaps, a benefit to those suffering from what might turn out to be OCD or an Anxiety disorder.

## Further Steps

There are three steps that stand out as highest priority:

    1. Increasing accuracy of the model by training the model on more data, analyzing misclassifications more carefully and implementing more preprocessing tools like lemmatization.

    2. Getting input from users and professionals. it's hard to think of any useful tool or service that should be developed by a single datascientist. Instead, tools hsould be developed with input from the actual users as well as experts in the field.

    3. Further applications: Ideally this is just a start and future tools could analyze free text comments from users in subreddits dedicated to other mental health disorders.

# Souces:
Three sources were used in our presentation and the pushshift.io api was used to get the reddit.com data.
1. https://www.mayoclinic.org/diseases-conditions/anxiety/symptoms-causes/syc-20350961
2. https://www.mayoclinic.org/diseases-conditions/obsessive-compulsive-disorder/symptoms-causes/syc-20354432
3. https://www.healthychildren.org/English/health-issues/conditions/emotional-problems/Pages/Anxiety-Disorders.aspx
