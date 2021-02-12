import nltk
import numpy as np
import pandas as pd
import pickle
import random
import pprint, time
from sklearn.model_selection import train_test_split
from sklearn_crfsuite import metrics
from sklearn_crfsuite import scorers
from collections import Counter
from .utils import *

class POS_HMM:
    #def __init__(self,):
    
    def HMM_Viterbi(self,words, train_bag,tags_df):
        state = []
        T = list(set([pair[1] for pair in train_bag]))
        
        for key, word in enumerate(words):
            #initialise list of probability column for a given observation
            p = [] 
            for tag in T:
                if key == 0:
                    transition_p = tags_df.loc['<e>', tag]
                else:
                    transition_p = tags_df.loc[state[-1], tag]
                    
                # compute emission and state probabilities
                emission_p = word_given_tag(words[key], tag, train_bag)[0]/word_given_tag(words[key], tag, train_bag)[1]
                state_probability = emission_p * transition_p    
                p.append(state_probability)
                
                
            pmax = max(p)
            if pmax==0:
                state_max=rules_key(words[key])
            else:
                # getting state for which probability is maximum
                state_max = T[p.index(pmax)] 
            state.append(state_max)
            
        return list(state)


    def train(self, input="Dataset/Augmented_POSTAG_training_ml.csv", test_size=0.05):
        
        df=pd.read_csv(input)
        tagged_sentence=Preparing_tagged_data(df)
        #printing details
        printing_details(tagged_sentence)
        
        train_set, test_set = train_test_split(tagged_sentence,train_size=0.95,test_size=test_size,random_state=7)
        
        
        train_tagged_words = [ tup for sent in train_set for tup in sent ]
        test_tagged_words = [ tup for sent in test_set for tup in sent ]
        
        
        vocab = {word for word,tag in train_tagged_words}
        tags = {tag for word,tag in train_tagged_words}
        
        
        # creating t x t transition matrix of tags, t= no of tags
        # Matrix(i, j) represents P(jth tag after the ith tag)
        tags_df=TransitionMatrix(tags,train_tagged_words)
        print(tags_df)
        
        print("Saving Transition Matrix.....")
        # Save the Model to file in the current working directory
        
        tags_df.to_csv('Saved_Models/POS/TransitionMatrix_HMM.csv')    
        print("File Saved at " + 'Saved_Models/POS/TransitionMatrix_HMM.csv')
        print()    
        
        print("Checking the Algoritham's Performance \n")
        self.test(test_set,train_tagged_words,tags_df)
        

    
    def test(self,test_set,train_bag,tags_df):
        start = time.time()
        y_test=[]
        y_pred=[]
        for sent in test_set:
            y_test_untagged = [word[0] for word in sent]

            y_test.append([word[1] for word in sent])

            tagged_seq = self.HMM_Viterbi(y_test_untagged,train_bag,tags_df)

            y_pred.append(tagged_seq)

        end = time.time()
        difference = end-start
        
        print("Time taken in seconds: ", difference)


        print("test accuracy is %f \n" % np.multiply(metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=None),100))
        print("Test classification report is \n")
        print(metrics.flat_classification_report(y_test, y_pred, digits=3,labels=None))

    def Predict_Data(self,Monolingual_sumerian,train_bag,tags_df):
        start = time.time()
        y_pred=[]
        for i in range(len(Monolingual_sumerian)):
            l=Monolingual_sumerian[i].split()
            tagged_seq = self.HMM_Viterbi(l,train_bag,tags_df)
            y_pred.append(tagged_seq)

        end = time.time()
        difference = end-start
        print("Time taken in seconds: ", difference)
        
        return y_pred
    
    def predict(self, input="Dataset/sumerian_demo.txt", 
                output='Output/POS_HMM.txt',):
       
        print("\n")
        print("Input file is ", input)
        print("Output file will be ", output)
        print("\n")

        Monolingual_sumerian=Openfile(input)
    
        df=pd.read_csv('Dataset/Augmented_POSTAG_training_ml.csv')
        tagged_sentence=Preparing_tagged_data(df)
        
        train_set, test_set = train_test_split(tagged_sentence,train_size=0.90,test_size=0.10,random_state=42)
        
        
        train_tagged_words = [ tup for sent in train_set for tup in sent ]
        test_tagged_words = [ tup for sent in test_set for tup in sent ]
        
        
        vocab = {word for word,tag in train_tagged_words}
        tags = {tag for word,tag in train_tagged_words}
        
        # creating t x t transition matrix of tags, t= no of tags
        # Matrix(i, j) represents P(jth tag after the ith tag)
        tags_df=TransitionMatrix(tags,train_tagged_words)
        
        Prediction=self.Predict_Data(Monolingual_sumerian,train_tagged_words,tags_df)
        POS_list=POSLIST(Monolingual_sumerian,Prediction)
        print("Saving_file "+output)
        Savefile(output, Monolingual_sumerian,POS_list)