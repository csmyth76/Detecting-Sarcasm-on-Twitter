from __future__ import unicode_literals
import re
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from wordcloud import WordCloud
import matplotlib.pyplot as plt 
import pandas as pd 
import seaborn as sns
import numpy as np
from sklearn.manifold import TSNE
from textblob import TextBlob 
import operator
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import emoji
from emoji.unicode_codes import UNICODE_EMOJI
import spacy,en_core_web_sm
import textacy
from textblob import TextBlob 
from IPython.display import clear_output


'''   
# use to import this file in jupyter without 
# restarting the kernel

import importlib
import data_utility as du
importlib.reload(du)
'''


'''

Usage Example:

ridgeParams = {'C':np.linspace(.000149,.0075,50),
              'penalty':('l2',)}
lrlrd = GridSearchCV(LogisticRegression(), 
                     ridgeParams, 
                     return_train_score=True, 
                     verbose=True)
gslrd = lrlrd.fit(X,y)
GridSearchTablePlot(gslrd,"C")

'''

def GridSearchTablePlot(gridClf, paramName,
                          num_results=15,
                          negative=False,
                          graph=True,
                          displayAllParams=False,
                          largeTable=False):

    clf = gridClf.best_estimator_
    clf_params = gridClf.best_params_
    if negative:
        clf_score = -gridClf.best_score_
    else:
        clf_score = gridClf.best_score_
    clf_stdev = gridClf.cv_results_['std_test_score'][gridClf.best_index_]
    cv_results = gridClf.cv_results_

    print("best parameters: {}".format(clf_params))
    print("best score:      {:0.5f} (+/-{:0.5f})".format(clf_score, clf_stdev))
    if displayAllParams:
        import pprint
        pprint.pprint(clf.get_params())

    # pick out the best results
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')

    best_row = scores_df.iloc[0, :]
    if negative:
        best_mean = -best_row['mean_test_score']
    else:
        best_mean = best_row['mean_test_score']
    best_stdev = best_row['std_test_score']
    best_param = best_row['param_' + paramName]

 
    # display abreviated top 'num_results' results
    if largeTable:
        display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results))

    else:
        abrevCols = ['mean_test_score','std_test_score']
        abrevCols += [col for col in pd.DataFrame(cv_results) if 'param_' in col]
    
        display(pd.DataFrame(cv_results) \
            .sort_values(by='rank_test_score').head(num_results)[abrevCols])
        

    # plot the results
    [col for col in pd.DataFrame(cv_results) if 'param_' in col]
    
    scores_df = scores_df.sort_values(by='param_' + paramName)

    if negative:
        means = -scores_df['mean_test_score']
    else:
        means = scores_df['mean_test_score']
    stds = scores_df['std_test_score']
    params = scores_df['param_' + paramName]

    # plot
    if graph:
        plt.figure(figsize=(8, 4))
        plt.errorbar(params, means, yerr=stds)

        plt.axhline(y=best_mean + best_stdev, color='red')
        plt.axhline(y=best_mean - best_stdev, color='red')
        plt.plot(best_param, best_mean, 'or')

        plt.title(paramName + " vs Score\nBest Score {:0.5f}".format(clf_score))
        plt.xlabel(paramName)
        plt.ylabel('Score')
        plt.show()
        
        
        
'''
Usage Example:

knnModelParams = {'n_neighbors': 4,
               'weights': 'distance'}
knnModel = Model('KNN',X,y,KNeighborsClassifier(),**knnModelParams)
knnModel.performance()
'''

import time
from sklearn.model_selection import cross_val_score

class Model(object):
    
    def __init__(self,name,X,Y,modelObj,**modelParams):
        self.params = modelParams
        self.modelObj = modelObj.set_params(**self.params)  
        self.name = name
        self.X = X
        self.Y = Y
        self.initValues()   
    
    def initValues(self):
        start = time.time()
        self.modObjFit = self.modelObj.fit(self.X, self.Y)
        end = time.time()
        self.timeToFit = end - start
        #self.crossVal = cross_val_score(self.modelObj, self.X, self.Y, cv=5, scoring='roc_auc')
        self.crossVal = cross_val_score(self.modelObj, self.X, self.Y, cv=5)
        self.crossValMean = np.mean(self.crossVal)
        self.crossValRange = max(self.crossVal) - min(self.crossVal)
         
    def performance(self, boxPlot=True):
        print ("Model:\t\t" + str(self.name))
        print ("CV Mean:\t" + str(self.crossValMean))
        print ("CV Range:\t" + str(self.crossValRange))
        print ("Train Time:\t" + str(self.timeToFit))
        print ("CV Scores: ")
        print (self.crossVal)
        if boxPlot:
            fig = plt.figure()
            title = "Performance: %s" % (self.name)
            fig.suptitle(title)
            ax = fig.add_subplot(111)
            plt.boxplot(self.crossVal, showmeans=True)
            ax.set_xticklabels(self.name)
            plt.show()
    
    def compareBox(self,modelList,filterResult=0, newTitle=''):
        results = []
        names = []
        
        results.append(self.crossVal)
        names.append(self.name)
        
        for rightModel in modelList:
            if filterResult > 0:
                if rightModel.crossVal.mean() > filterResult:
                    results.append(rightModel.crossVal)
                    names.append(rightModel.name)
            else:
                results.append(rightModel.crossVal)
                names.append(rightModel.name)
        
        fig = plt.figure()
        if newTitle != '':
            title = newTitle
        else:
            title = "Performance: %s" % (self.name)
        fig.suptitle(title)
        ax = fig.add_subplot(111)
        
        plt.boxplot(results, showmeans=True)
        ax.set_xticklabels(names)
        plt.setp(ax.get_xticklabels(), rotation=90)
        plt.show()
        
    def fullCompareList(self,listOfModels,folds=False):
        listOfModels.insert(0,self)
        dfout = pd.DataFrame()
        dataForFrame = []
        cvNameList= []
        for model in listOfModels:
            dataRowForFrame = {}
            dataRowForFrame = {'Name': model.name,
                                'Time':model.timeToFit,
                                'CV Mean':model.crossValMean,
                                'CV Range': model.crossValRange}
            if folds:  
                buildCvNameList = False
                if not cvNameList:
                    buildCvNameList=True
                for idx,cv in enumerate(model.crossVal):
                    cvName = 'CV Fold ' + str(idx+1)
                    if buildCvNameList:
                        cvNameList.append(cvName)
                    dataRowForFrame[cvName] = cv
            dataForFrame.append(dataRowForFrame)
        colOrder = ['Name', 'CV Mean', 'CV Range', 'Time']
        colOrder += cvNameList  
        dfOut = pd.DataFrame(dataForFrame)
        dfOut = dfOut[colOrder]
        display(dfOut)
        lgrbg = listOfModels.pop(0)
        self.compareBox(listOfModels)    
        
        
 # from wikipedia

contraction_dict = {
  "ain't": "am not",
  "aren't": "are not",
  "can't": "cannot",
  "can't've": "cannot have",
  "'cause": "because",
  "could've": "could have",
  "couldn't": "could not",
  "couldn't've": "could not have",
  "didn't": "did not",
  "doesn't": "does not",
  "don't": "do not",
  "hadn't": "had not",
  "hadn't've": "had not have",
  "hasn't": "has not",
  "haven't": "have not",
  "he'd": "he would",
  "he'd've": "he would have",
  "he'll": "he will",
  "he'll've": "he will have",
  "he's": "he is",
  "how'd": "how did",
  "how'd'y": "how do you",
  "how'll": "how will",
  "how's": "how is",
  "I'd": "I would",
  "I'd've": "I would have",
  "I'll": "I will",
  "I'll've": "I will have",
  "I'm": "I am",
  "I've": "I have",
  "isn't": "is not",
  "it'd": "it had",
  "it'd've": "it would have",
  "it'll": "it will",
  "it'll've": "it will have",
  "it's": "it is",
  "let's": "let us",
  "ma'am": "madam",
  "mayn't": "may not",
  "might've": "might have",
  "mightn't": "might not",
  "mightn't've": "might not have",
  "must've": "must have",
  "mustn't": "must not",
  "mustn't've": "must not have",
  "needn't": "need not",
  "needn't've": "need not have",
  "o'clock": "of the clock",
  "oughtn't": "ought not",
  "oughtn't've": "ought not have",
  "shan't": "shall not",
  "sha'n't": "shall not",
  "shan't've": "shall not have",
  "she'd": "she would",
  "she'd've": "she would have",
  "she'll": "she will",
  "she'll've": "she will have",
  "she's": "she is",
  "should've": "should have",
  "shouldn't": "should not",
  "shouldn't've": "should not have",
  "so've": "so have",
  "so's": "so is",
  "that'd": "that would",
  "that'd've": "that would have",
  "that's": "that is",
  "there'd": "there had",
  "there'd've": "there would have",
  "there's": "there is",
  "they'd": "they would",
  "they'd've": "they would have",
  "they'll": "they will",
  "they'll've": "they will have",
  "they're": "they are",
  "they've": "they have",
  "to've": "to have",
  "wasn't": "was not",
  "we'd": "we had",
  "we'd've": "we would have",
  "we'll": "we will",
  "we'll've": "we will have",
  "we're": "we are",
  "we've": "we have",
  "weren't": "were not",
  "what'll": "what will",
  "what'll've": "what will have",
  "what're": "what are",
  "what's": "what is",
  "what've": "what have",
  "when's": "when is",
  "when've": "when have",
  "where'd": "where did",
  "where's": "where is",
  "where've": "where have",
  "who'll": "who will",
  "who'll've": "who will have",
  "who's": "who is",
  "who've": "who have",
  "why's": "why is",
  "why've": "why have",
  "will've": "will have",
  "won't": "will not",
  "won't've": "will not have",
  "would've": "would have",
  "wouldn't": "would not",
  "wouldn't've": "would not have",
  "y'all": "you all",
  "y'alls": "you alls",
  "y'all'd": "you all would",
  "y'all'd've": "you all would have",
  "y'all're": "you all are",
  "y'all've": "you all have",
  "you'd": "you had",
  "you'd've": "you would have",
  "you'll": "you you will",
  "you'll've": "you you will have",
  "you're": "you are",
  "you've": "you have"
}


'''
PARAMETERS:
List of sentences with contractions

RETURN:
List from input parameter but with contractions expanded

USAGE:
cont_text = ['you\'re ain\'t not', 'you\'ve really, you\'ll' ]
expand_contractions(cont_text)

'''
def expand_contractions(text_list):
    pattern = re.compile(r'\b(' + '|'.join(contraction_dict.keys()) + r')\b')

    contractions_expanded = []
    for t in text_list:
        # convert twitter apostrophes to apostrophes that can be detected in the pattern
        t = t.replace('’','\'')
        # expand contraction
        t = pattern.sub(lambda x: contraction_dict[x.group()], t.lower())
        # remove possesives 
        t = t.replace('\'s', '')
        contractions_expanded.append(t)  
    return contractions_expanded


def get_tweet_processor(additional_dictionary_list = None):
    
    dicts = [emoticons]
    #print (dicts)
    print (len(dicts))
    if additional_dictionary_list:
        dicts.extend(additional_dictionary_list)

    print (len(dicts))
    
    '''
    Test with this code block:

    sentences = [
    "he's aaaaaaaaand rt CANT WAIT for the ijwts new season of #TwinPeaks ＼(^o^)／!!! #davidlynch #tvseries :)))",
    "I saw the new #johndoe movie and it suuuuucks!!! WAISTED $10... #badmovies :/",
    "@SentimentSymp:  can't wait for the Nov 9 #Sentiment talks!  YAAAAAAY !!! :-D http://sentimentsymposium.com/."
    ]

    for s in sentences:
    print(" ".join(text_processor.pre_process_doc(s)))

    '''

    text_processor = TextPreProcessor(
    # terms that will be normalized
    normalize=['email', 'percent', 'money', 'phone', 'user',
        'time', 'url', 'date', 'number'],
    # terms that will be annotated
    annotate={"hashtag", "allcaps", "elongated", "repeated",
        'emphasis', 'censored'},
    fix_html=True,  # fix HTML tokens

    # corpus from which the word statistics are going to be used 
    # for word segmentation 
    segmenter="twitter", 

    # corpus from which the word statistics are going to be used 
    # for spell correction
    corrector="twitter", 

    unpack_hashtags=True,  # perform word segmentation on hashtags
    unpack_contractions=True,  # Unpack contractions (can't -> can not)
    spell_correct_elong=True,  # spell correction for elongated words

    # select a tokenizer. You can use SocialTokenizer, or pass your own
    # the tokenizer, should take as input a string and return a list of tokens
    tokenizer=SocialTokenizer(lowercase=True).tokenize,

    # list of dictionaries, for replacing tokens extracted from the text,
    # with other expressions. You can pass more than one dictionary.
    dicts=dicts
    )
    
    return text_processor




''' 
remove punctuation, numbers and special characters

tweet: tweet string of text 

Return: same input string without punctuation, numbers and special characters
'''
def clean_tweet(tweet): 
    return ' '.join(re.sub("([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split()) 


def lemmatize_list(text_list):
    lmt = WordNetLemmatizer()

    lemmatized = []
    for tweet in text_list:
        tokenized_words = word_tokenize(tweet)
        lemmatized_tweet = []
        for word in tokenized_words:
            lemmatized_word = lmt.lemmatize(word)
            lemmatized_tweet.append(lemmatized_word)
        lemmatized.append(' '.join(lemmatized_tweet))
    
    return lemmatized
  

    
def remove_tags_from_processed_text(text_list):

    tags_to_remove = [
        # leaving the processed text of the hashtag in the text but removing the tags
        # ex. '#toogood' becomes '<hashtag> too good </hashtag>' ends as 'too good')
        '<hashtag>',
        '</hashtag>',
        # drop these tags
        '<allcaps>',
        '</allcaps>',
        '<happy>',
        '<elongated>',
        '<repeated>',
        '<emphasis>',
        '<email>',
        '<percent>',
        '<money>',
        '<phone>',
        '<user>',
        '<time>',
        '<number>',
        '<date>',
        '<wink>',
        '<laugh>',
        '<censored>',
        '<sad>',
        '<annoyed>',
        '<tong>',
        '<url>']
    for t in tags_to_remove:
        text_list = text_list.str.replace(t, '')
    return text_list;
       
    
def sentences_to_words(sentence_list, drop_words=None):
    
    words = ' '.join([sent_text for sent_text in sentence_list])
    if drop_words:
        for wrd in drop_words:
            # replace only the exact word
            words = re.sub(r"\b%s\b" % wrd , 
               r"%s" % '', 
               words, 
               flags=re.IGNORECASE)
            # get rid of double spaces left from sub
            words = words.replace('  ',' ')
    return words
        

    
def sentences_to_wordcloud(sentence_list, drop_words_list=None):
    
    words = sentences_to_words(sentence_list, drop_words_list)
    
    wordcloud = WordCloud(width=800, 
                          height=500, 
                          random_state=21, 
                          collocations=False,
                          max_font_size=110).generate(words)

    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show() 
    

    
    
from operator import itemgetter

def sentences_to_sortedwordcount(sentence_list, drop_words_list=None):
    
    words = sentences_to_words(sentence_list, drop_words_list)
    counts = {}
    words = words.split()

    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    
    sortedwords = sorted(counts.items(), key=itemgetter(1), reverse = True)
     
    return sortedwords



def wordcount_to_bar(x_label, y_label, word_count_tuple, rotate=False):
    a = dict(word_count_tuple)
    d = pd.DataFrame({x_label: list(a.keys()),
                      y_label: list(a.values())})
    # selecting top 10 most frequent hashtags     
    d = d.nlargest(columns=y_label, n = 10) 
    plt.figure(figsize=(16,5))
    if rotate:
        plt.xticks(rotation = 45)
    ax = sns.barplot(data=d, x=x_label, y=y_label)
    ax.set(ylabel=y_label)
    plt.show()


def hashtag_extract(x):
    hashtags = []
    # Loop over the words in the tweet
    for i in x:
        ht = re.findall(r"#(\w+)", i)
        ht = ' '.join(ht)
        if ht != '':
            hashtags.append(ht)

    return hashtags    
 
    
def sentences_to_sorted_hashtags(sentence_list, drop_words=None):
    ht_only = hashtag_extract(sentence_list)
    return sentences_to_sortedwordcount(ht_only, drop_words)    


def build_corpus(data):
    "Creates a list of lists containing words from each tweet"
    corpus = []
    for sentence in data['text'].iteritems():
        word_list = sentence[1].split(" ")
        corpus.append(word_list)       
    return corpus


def tsne_plot(model):
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=2500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()



'''
Parameters:
compare_data_name - Name of the date that being compared; will be used in graph title
compare_data_list - Data being compared
label_list - List of 0 or 1 labels that indicates a sarcastic vs non-sarcastic tweet
'''
def sarc_vs_non_graph(compare_data_name, compare_data_list, label_list):
    s_v_n_ds = pd.DataFrame()
    text_label = ['sarcastic' if l == 1  else 'non-sarcastic' for l in label_list]
    title = 'Sarcastic vs Nonsarcastic: ' + compare_data_name
    s_v_n_ds['Sarcasm'] = text_label
    s_v_n_ds[compare_data_name] = compare_data_list                                 
    sns.set(style="whitegrid")
    g = sns.catplot(x=compare_data_name,
        hue='Sarcasm',
        data=s_v_n_ds,kind='count')
    plt.title(title)
   

'''
Same as sarc_vs_non_graph but provides the ability to filter the compare_data_list
based on the counts (i.e. only see the graph for tweets that have more that 0 hashtags and less than 12).  
'''
def sarc_vs_non_graph_limit(compare_data_name, 
                            compare_data_list, 
                            label_list,
                            lower_limit,
                            upper_limit = 0):
    s_v_n_l_ds = pd.DataFrame()
    s_v_n_l_ds['Sarcasm'] = label_list
    s_v_n_l_ds[compare_data_name] = compare_data_list
    s_v_n_l_ds = s_v_n_l_ds[s_v_n_l_ds[compare_data_name] > lower_limit].reset_index(drop=True)
    if upper_limit:
        s_v_n_l_ds = s_v_n_l_ds[s_v_n_l_ds[compare_data_name] < upper_limit].reset_index(drop=True)
        
    du.sarc_vs_non_graph(compare_data_name, 
                            s_v_n_l_ds[compare_data_name], 
                            s_v_n_l_ds['Sarcasm'])

    

'''
EXAMPLE:
text=['This@ and this@and THIS', 'this and Thatthis @']
label=[1,0]
word_count_sarc_nonsarc('this', text, label)
#label 1 = 3
#label 0 = 2
'''
def word_count_sarc_nonsarc(find_word, text, label):
    label_1_count = 0
    label_0_count = 0
    for index, row in enumerate(text):
        if find_word.lower() in row.lower():

            if label[index]==1:
                label_1_count+=row.lower().count(find_word.lower())
            if label[index]==0:
                label_0_count+=row.lower().count(find_word.lower())

    print ('label 1 = {}'.format(label_1_count))    
    print ('label 0 = {}'.format(label_0_count))     
    

def word_count_sarc_nonsarc_list(find_word_list, text, label):
    
    for word in find_word_list:
        print (word)
        word_count_sarc_nonsarc(word, text, label)

'''
EXAMPLE:

text=['This@ and this@and THIS', 'this and Thatthis @','one two three','this one as well']
label=[1,0,0,1]
word_usage_sarc_nonsarc('this', text, label)
#label 1
#  This@ and this@and THIS
#  this one as well
#label 0
#  this and Thatthis @
'''
def word_usage_sarc_nonsarc(find_word, text, label):
    label_1 = []
    label_0 = []
    for index, row in enumerate(text):
        if find_word.lower() in row.lower():
            if label[index]==1:
                label_1.append(row)
            if label[index]==0:
                label_0.append(row)
    print ('label 1')
    for sent_1 in label_1:
        print ('-'+sent_1) 
    print ('label 0') 
    for sent_0 in label_0:
        print ('-'+sent_0)


def word_usage_sarc_nonsarc_list(find_word_list, text, label):
    
    for word in find_word_list:
        print (word)
        word_usage_sarc_nonsarc(word, text, label)

        
'''
Used to remove a single word from a dataset.

Return:
Input list without remove_word in any of the list elements.
'''
def remove_word_from_list(remove_word, text_list):
    list_no_word = []
    ignorecase_remove_word = re.compile(re.escape(remove_word), re.IGNORECASE)
    for row in text_list:
        list_no_word.append(ignorecase_remove_word.sub('',row))
    return list_no_word


def get_tweet_sentiment(tweet): 
    ''' 
    Utility function to classify sentiment of passed tweet 
    using textblob's sentiment method 
    '''
    # create TextBlob object of passed tweet text 
    analysis = TextBlob(tweet) 
    # set sentiment 
    if analysis.sentiment.polarity > 0: 
        return 'positive'
    elif analysis.sentiment.polarity == 0: 
        return 'neutral'
    else: 
        return 'negative'

    
def get_tweet_polarity_subjectivity(tweet): 
    ''' 

    '''
    pol_and_sub = {} 
    analysis = TextBlob(tweet) 
    # set sentiment 
    pol_and_sub['polarity'] = analysis.sentiment.polarity
    pol_and_sub['subjectivity'] = analysis.sentiment.subjectivity
    
    return pol_and_sub


def get_sentiment_textblob(text_list):
    sentiment = []
    for t in text_list:
        sentiment.append(get_tweet_sentiment(t))
    return sentiment


def sentiment_intensity_box_compare(title, intensity_element_name, sent_feats_df, label_list):
    names = ['Non-sarcastic', 'Sarcastic']
    sent_feats_df['label'] = label_list
    results = [sent_feats_df[sent_feats_df['label']==0][intensity_element_name], 
               sent_feats_df[sent_feats_df['label']==1][intensity_element_name]]

    fig = plt.figure()
    fig.suptitle(title)
    ax = fig.add_subplot(111)

    plt.boxplot(results, showmeans=True)
    ax.set_xticklabels(names)
    plt.setp(ax.get_xticklabels(), rotation=45)
    plt.show()  


def counts_to_graph(count_item_name, count_data):
    graph_data = pd.DataFrame()
    graph_data[count_item_name] = count_data
    sns.set(style="whitegrid")
    g = sns.catplot(x=count_item_name,
        data=graph_data,kind='count')
    

def counts_to_graph_filter(none_below, count_item_name, count_data):
    filtered_usage_count = [c for c in count_data if c > none_below]
    counts_to_graph(count_item_name, filtered_usage_count)
    
    
def get_tweets_based_on_count(count_to_get, tweets_list, count_list):
    count_df = pd.DataFrame()
    count_df['text'] = tweets_list
    count_df['counts'] = count_list
   
    return count_df[count_df['counts'] == count_to_get]['text']
    
    
'''
takes the tagged output from TextPreProcessor and provides a dictionary
with the elongated and repated tag counts per line
'''
def get_elongated_repeated_from_tags(processed_text_list):
    elongated = []
    repeated = []
    for tweet in processed_text_list:
        elongated.append(tweet.count('<elongated>'))
        repeated.append(tweet.count('<repeated>'))
    return {'elongated': elongated, 'repeated':repeated}
 

'''
takes the tagged output from TextPreProcessor and provides a dictionary
with the elongated and repated tag counts per line
'''
def get_hashtag_mention_counts(processed_text_list):
    hashtags = []
    mentions = []
    for tweet in processed_text_list:
        hashtags.append(tweet.count('#'))
        mentions.append(tweet.count('@'))
    return {'hashtags': hashtags, 'mentions':mentions}    
       
    
def get_tfidf_score(score_word, tfidf_vect):
    t_idx = tfidf_vect.vocabulary_[score_word]
    return (tfidf_vect.idf_[t_idx])


def get_tfidf_scores_dict(tfidf_vect):
    idf_dict={}
    for wrd,idx in tfidf_vect.vocabulary_.items():
        idf_val = tfidf_vect.idf_[idx]
        idf_dict[wrd] = idf_val
    return idf_dict

def get_tfidf_scores_sorted(tfidf_vect):
    idf_dict = get_tfidf_scores_dict(tfidf_vect)
    sorted_idf_list = sorted(idf_dict.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_idf_list



def v_sentiment_scores(sentence):
    analyser = SentimentIntensityAnalyzer()
    snt = analyser.polarity_scores(sentence)
    return snt

def v_phrase_scores(sentence=None, phrase_type='verb'):

    if (phrase_type == 'verb'):
        pattern = r'<VERB>?<ADV>*<VERB>+'
    elif(phrase_type == 'noun'):
        pattern = r'<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+'
    else:
        print ('Unregognized phrase type')
        return
    
    doc = textacy.Doc(sentence, lang='en_core_web_sm')
    lists = textacy.extract.pos_regex_matches(doc, pattern)

    tot_phrase_score = {}
    tot_phrase_score['pos'] = 0
    tot_phrase_score['neg'] = 0
    tot_phrase_score['neu'] = 0
    tot_phrase_score['compound'] = 0
    for list in lists:
        phrase_score = v_sentiment_scores(list.text)
        tot_phrase_score['pos'] += phrase_score['pos']
        tot_phrase_score['neg'] += phrase_score['neg']
        tot_phrase_score['neu'] += phrase_score['neu']
        tot_phrase_score['compound'] += phrase_score['compound']
    return tot_phrase_score

import nltk 
from nltk.corpus import wordnet 

def get_antonyms_for_word (word):
    '''
    The lemmas will be synonyms, and then use .antonyms 
    to find the antonyms to the lemmas. 
    '''
    # get tags
    # check for antoyms
    antonyms = []
    synonyms = []
    for syn in wordnet.synsets(word): 
        for l in syn.lemmas(): 
            synonyms.append(l.name()) 
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name().replace('_', ' ')) 

    if antonyms:
        antonyms = set(antonyms)
    return antonyms


def get_antonyms_for_sentence(sentence):
    all_antonyms = []
    
    for wd in sentence.split():
        antonyms = get_anonyms_for_word(wd)
        all_antonyms += antonyms

    if all_antonyms:
        all_antonyms = set(all_antonyms)
    return all_antonyms


def antonym_detection (sentence):
    antonyms = get_anonyms_for_sentence(sentence)
    for ant in antonyms:
        if ant in sentence:
            return 1
    return 0


def generate_sentiment_feats(text_list):
    
    analyser = SentimentIntensityAnalyzer()

    ft_neg = []
    ft_pos = []
    ft_neu = []
    ft_cmpd = []
    vp_neg = []
    vp_pos = []
    vp_neu = []
    vp_cmpd = []
    n_neg = []
    n_pos = []
    n_neu = []
    n_cmpd = []
    sent_and_pos_feats = pd.DataFrame()

    for t in text_list:
        
        v_scores_ft = analyser.polarity_scores(t)
        ft_neg.append(v_scores_ft['neg'])
        ft_pos.append(v_scores_ft['pos'])
        ft_neu.append(v_scores_ft['neu'])
        ft_cmpd.append(v_scores_ft['compound'])

        v_scores_vp = v_phrase_scores(t, 'verb')
        vp_neg.append(v_scores_vp['neg'])
        vp_pos.append(v_scores_vp['pos'])
        vp_neu.append(v_scores_vp['neu'])
        vp_cmpd.append(v_scores_vp['compound'])

        v_scores_n = v_phrase_scores(t, 'noun')
        n_neg.append(v_scores_n['neg'])
        n_pos.append(v_scores_n['pos'])
        n_neu.append(v_scores_n['neu'])
        n_cmpd.append(v_scores_n['compound'])
      
    sent_and_pos_feats['t_neg'] = ft_neg
    sent_and_pos_feats['t_pos'] = ft_pos
    sent_and_pos_feats['t_neu'] = ft_neu
    sent_and_pos_feats['t_cmpd'] = ft_cmpd

    sent_and_pos_feats['vp_neg'] = vp_neg
    sent_and_pos_feats['vp_pos'] = vp_pos
    sent_and_pos_feats['vp_neu'] = vp_neu
    sent_and_pos_feats['vp_cmpd'] = vp_cmpd

    sent_and_pos_feats['n_neg'] = n_neg
    sent_and_pos_feats['n_pos'] = n_pos
    sent_and_pos_feats['n_neu'] = n_neu
    sent_and_pos_feats['n_cmpd'] = n_cmpd
    
    return(sent_and_pos_feats)




def extract_emojis(text):
    return ''.join(c for c in text if c in emoji.UNICODE_EMOJI)


'''
Emojis to test with:
'🤔 🙈 me así,🤣 bla😂😂 es 👏se 😌 ds 💕👭👙'
'''

# collect and count emoticons
def get_emoji_from_text(text):
    emoji_list = []
    for character in text:
        if character in emoji.UNICODE_EMOJI:
            emoji_list.append(character)
    return emoji_list


def get_emoji_usage_count(text_list):
    emoji_count = []
    for row in text_list:
        emoji_count.append(len(get_emoji_from_text(row)))
    return emoji_count


'''
EXAMPLE:
sents = ['🤔 🙈 me así,🤣 bla😂', '😂 es 👏se 😌 ds 💕👭👙', '😂 es 👏se']
get_sorted_emoji_counts(sents)
'''
def get_sorted_emoji_counts(text_list):
    collected_emojis_and_counts = {}
    for text in text_list:
        text_emoji_list = get_emoji_from_text(text)
        for one_emji in text_emoji_list:
            if one_emji in collected_emojis_and_counts:
                collected_emojis_and_counts[one_emji]+=1
            else:
                collected_emojis_and_counts[one_emji] = 1
    sortedemojis = sorted(collected_emojis_and_counts.items(), key=itemgetter(1), reverse = True)
    
    return sortedemojis


'''
Parameters:
emoji_count_list - output from get_sorted_emoji_counts

Returns:
list of emoji tuples (text, count)
'''
def demojize_sorted_emoji_counts(emoji_count_list):
    updated_emoji_list = []
    for emoji_count in emoji_count_list:
        emoji_as_text = emoji.demojize(emoji_count[0])
        emoji_tup = (emoji_as_text, emoji_count[1])
        updated_emoji_list.append(emoji_tup)
    return updated_emoji_list



'''
Parameters:
text_list - list of tweets
cutoff - any emoji's with a count lower than this will not be 
    turned into a feature

Returns:
Features that can be appened to the provided data set
'''
def get_bag_of_emoji(text_list, cutoff=10):
    emoji_count_list = get_sorted_emoji_counts(text_list)
    list_of_tuple_counts = demojize_sorted_emoji_counts(emoji_count_list)

    feat_list = []
    for tuple_count in list_of_tuple_counts:
        if tuple_count[1]>= cutoff:
            feat_list.append(tuple_count[0])
    #initialize a dictionary of lists; each list is an emoji
    feat_col={e: [] for e in feat_list}
    for text in text_list:
        for single_feat in feat_list:
            feat_col[single_feat].append(emoji.demojize(text).count(single_feat))

    return pd.DataFrame.from_dict(feat_col)


def chunk_and_sentiment (sentence):

    nlp = en_core_web_sm.load()
    tb_out = TextBlob(sentence)
    print(tb_out.tags)
    pattern = r'<VERB>?<ADV>*<VERB>+'
    noun_phrase = r'<DET>? (<NOUN>+ <ADP|CONJ>)* <NOUN>+'
    compound_nouns = r'<NOUN>+'
    verb_phrase = r'<VERB>?<ADV>*<VERB>+'
    prepositional_phrase= r'<PREP> <DET>? (<NOUN>+<ADP>)* <NOUN>+'

    doc = textacy.Doc(sentence, lang='en_core_web_sm')
    print (sentence)
    #print (get_tweet_polarity_subjectivity(sentence))
    print (v_sentiment_scores(sentence))

    lists = textacy.extract.pos_regex_matches(doc, pattern)
    for list in lists:
        print('Verb phrase: ' +list.text)
        print (v_sentiment_scores(list.text))
        #print (get_tweet_polarity_subjectivity(list.text))
        verb_phrase = list.text
    print ('')    
    lists = textacy.extract.pos_regex_matches(doc, noun_phrase)
    for list in lists:
        print('Noun phrase: ' +list.text)
        print (v_sentiment_scores(list.text))
        #print (get_tweet_polarity_subjectivity(list.text))
        noun_phrase = list.text
    print ('')   
    lists = textacy.extract.pos_regex_matches(doc, compound_nouns)
    for list in lists:
        print('Compound noun: ' + list.text)
        print (v_sentiment_scores(list.text))
        #print (get_tweet_polarity_subjectivity(list.text))
        verb_phrase = list.text
    print ('')
    lists = textacy.extract.pos_regex_matches(doc, prepositional_phrase)
    for list in lists:
        print('Prepositional phrase: ' + list.text)
        print (v_sentiment_scores(list.text))
        #print (get_tweet_polarity_subjectivity(list.text))
        verb_phrase = list.text
    for word in sentence.split():
        print (word)
        print (v_sentiment_scores(word))