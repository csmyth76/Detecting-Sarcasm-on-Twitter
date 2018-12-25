# Detecting Sarcasm on Twitter

Project File: semanticRel.ipynb

## Introduction

Sarcasm is a mode of communication where the meaning of a statement is the opposite of what is said. Detecting sarcasm in written text could help determine the true intent or sentiment of individuals towards issues, products or services.

In verbal communication the presence of sarcasm can be conveyed through facial expressions and tone. While these tools are not available for written text, other modes of expression such as emoji, elongated text and repeated text are. The model explored these, other twitter specific forms of expression (hashtags and mentions) as well as sentiment.

## Summary

### Preparation:

Multiple applications and dictionaries had to be built in order to normalize twitter text due to extensive use of slang and acronyms

### General Data Exploration:

- Most tweets did not utilize emoji, or hashtags
- Most tweets had at least one Mention (18 tweets had 50 mentions)
- LDA analysis did reveal interpretable topics among the tweets

#### Sarcastic vs Non-Sarcastic Data Comparison:

- Very little difference in the most used words
- Very few words specific to sarcastic or non-sarcastic tweets
- Top non-sarcastic hashtags were pop culture related
- Both utilized the “face with tears of joy” emoji but sarcastic tweets utilized more of them
- When elongated words were used, both used mainly just 1 elongated word and both did it about as often
- Sarcastic tweets had more repeated text/emoji/punctuation counts
- Sarcastic tweets used hashtags almost three times more than non-sarcastic tweets
- Tweets that used an extreme number of hashtags (i.e more than 10) were almost all sarcastic
- Sarcastic tweets used slightly more mentions than non-sarcastic tweets
- Tweets that used an extreme number of mentions where mostly non-sarcastic
- Most tweets were positive with sarcastic tweets being slightly more positive and no-sarcastic tweets being slightly more negative
- When sentiment of noun phrase, verb phrase and the entire tweet were compared, most tweets did not have a contradiction in sentiment, however if there was a contradiction, sarcastic tweets contained slightly more sentiment contradictions than non-sarcastic

### Features:

- Bag of Emoji
- Repeated and Elongated word count
- Mention and Hashtag counts
- Detailed Sentiment of full tweet, verb phrases and noun phrases
- Contradiction flag: noted any contradiction between tweet, verb phrase and noun phrase
- Word level tf-idf

#### Most Important Features:

- Hashtag Count
- Tweet Neutral Sentiment Score
- Mention Count
- Noun Phrase Sentiment Score
- Full Tweet Sentiment Score

### Best Models:

- Vanilla Logistic Regression
- Ridge Regression
- Lasso Regression
- Extreme Gradient Boost
- Support Vector Machine

## Conclusion:

It was unexpected that the sarcastic and non-sarcastic tweets had so many similar characteristics. Given the nature of sarcasm I expected there to be more contradictions in sentiment between the tweet and the tweets verb phrases and noun phrases. If sentiment was going to be a factor, I expected it would have been the positive or negative sentiment that would have been the most influential. Instead it was the neutral sentiment scores that proved to be the most important features.

## Improvements:

- Larger dataset
- Loose antonyms
- Interjection detection

## Getting Tweets
A separate application has been built to get the data from twitter utilizing the twitter api. The application utilized the #sarcasm hashtag to identify sarcastic tweets.
