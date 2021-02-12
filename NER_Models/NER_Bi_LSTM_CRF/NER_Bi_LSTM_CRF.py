import numpy as np
import pandas as pd
import matplotlib
import pickle
from sklearn_crfsuite import metrics
from collections import OrderedDict 
from sklearn.model_selection import train_test_split
from keras.models import Sequential, load_model, Model, Input
from keras.layers import LSTM, Dense, Dropout, Embedding, Masking, Bidirectional,  TimeDistributed
from keras.optimizers import Adam
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy
from tqdm.keras import TqdmCallback
from .utils import *


class NER_Bi_LSTM_CRF:

    def __init__(self,n_LSTM=64, dropout=0.2, 
                recurrent_dropout=0.2, activation='relu', optimizer='adam' ):
        self.n_LSTM = n_LSTM
        self.dropout=dropout
        self.recurrent_dropout=recurrent_dropout
        self.activation=activation
        self.optimizer = optimizer

    
    def BUILD_MODEL(self,X,MAX,n_words,n_tags,embedding_matrix):
        input_word = Input(shape = (MAX,))
        model=Embedding(input_dim=n_words,input_length=X.shape[1], output_dim=embedding_matrix.shape[1], weights=[embedding_matrix],trainable=False)(input_word)
        model=Bidirectional(LSTM(self.n_LSTM, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(model)
        model=TimeDistributed(Dense(self.n_LSTM//2, activation =self.activation))(model)
        crf = CRF(n_tags)  # CRF layer
        out = crf(model)  # output
        model = Model(input_word, out)
        model.summary()
        model.compile(optimizer=self.optimizer, loss=crf.loss_function, metrics=[crf.accuracy, 'accuracy'])
        return model


    def preparedicts(self, df, embedding):
        vocab=list(df["FORM"].values)
        f = open(embedding)
        for line in f:
            values = line.split()
            word = values[0]
            vocab.append(word)
        vocab=sorted(list(set(vocab)))
        vocab.append("<end>")
        vocab.append("UNK")
        
        tags = sorted(list(set(df["XPOSTAG"].values)))
        tags.append("<e>")
        
        word2idx=OrderedDict() 
        idx2word=OrderedDict() 
        tag2idx=OrderedDict() 
        idx2tag=OrderedDict() 
        word2idx = {w: i for i, w in enumerate(vocab)}
        idx2word = {i: w for w, i in word2idx.items()}
        tag2idx = {t: i for i, t in enumerate(tags)}
        idx2tag = {i: w for w, i in tag2idx.items()}
        
        print("\n Saving Vocab as a list of json files \n")
        vocabulary=[word2idx,idx2word,tag2idx,idx2tag]
        with open('NER_Models/NER_Bi_LSTM_CRF/Sumerian_Vocab.pkl', 'wb') as f:
            pickle.dump(vocabulary,f)
            
        return word2idx,idx2word,tag2idx,idx2tag


    def train(self, MAX=50,input="Dataset/Augmented_NER_training_ml.csv",
              output="Saved_Models/NER/NER_Bi_LSTM_CRF.h5", 
              embedding='Word_Embeddings/sumerian_word2vec_50.txt',
              test_size=0.05, random_state=7):

        df= pd.read_csv(input)
        df=df[['ID','FORM','XPOSTAG']]
        print(df.head())
        sentences = Preparing_tagged_data(df)
        print ('Maximum sequence length:', MAX)
        word2idx,idx2word,tag2idx,idx2tag= self.preparedicts(df, embedding)
        X,y=prepareData(MAX,sentences,word2idx,tag2idx)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size,random_state=random_state)
    
        print("Dataset dimentions \n")
        print(X_train.shape,y_train.shape)
        print(X_test.shape,y_test.shape)
        embedding_matrix=embeddings(embedding, word2idx)
    
        model=self.BUILD_MODEL(X, MAX,len(word2idx),len(tag2idx),embedding_matrix)
    
    
        history = model.fit(X_train, y_train, epochs=10, batch_size=32,validation_split=0.1,verbose=0,callbacks=[TqdmCallback(verbose=2)])
        #evaluate_model(history)
        print("Saving model at ", output)
        model.save(output)
        self.test(model,X_test,y_test,idx2tag)


    def test(self,model,X_test,y_test,idx2tag,label=None):
        test_pred = model.predict(X_test, verbose=1)
        y_pred = pred2label(test_pred,idx2tag)
        y_test = pred2label(y_test,idx2tag)
        print("test accuracy is %f \n" % np.multiply(metrics.flat_f1_score(y_test, y_pred,average='weighted',labels=label),100))
        print("Test classification report is \n")
        print(metrics.flat_classification_report(y_test, y_pred, digits=3,labels=label))


    def predict(self,MAX = 50,input="Dataset/sumerian_demo.txt",
                saved="Saved_Models/NER/NER_Bi_LSTM_CRF.h5", 
                output='Output/NER_Bi_LSTM_CRF.txt'):

        Monolingual_sumerian=Openfile(input)
        loaded_model = load_model(saved,custom_objects={'CRF':CRF, 
                                                    'crf_loss':crf_loss, 
                                                    'crf_viterbi_accuracy':crf_viterbi_accuracy})
        
        word2idx,idx2word,tag2idx,idx2tag= preparedicts()
        X=preparetestData(MAX,Monolingual_sumerian,word2idx)

        Prediction=Predict_Testtag(loaded_model,X,Monolingual_sumerian,idx2tag)
        POS_list=POSLIST(Monolingual_sumerian,Prediction)

        print("Saving_file "+output)
        Savefile(Monolingual_sumerian,POS_list)
