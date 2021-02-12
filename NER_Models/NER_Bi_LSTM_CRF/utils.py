import numpy as np
import pickle
import matplotlib.pyplot as plt
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


## For Training
def Preparing_tagged_data(df):
    tagged_sentence=[]
    c=1
    temp=[]
    for i in range(len(df)):
        if df['ID'][i]==c:
            temp.append((df['FORM'][i],df['XPOSTAG'][i]))
        else:
            tagged_sentence.append(temp)
            temp=[]
            temp.append((df['FORM'][i],df['XPOSTAG'][i]))
            c+=1
    tagged_sentence.append(temp)
    return tagged_sentence



def prepareData(MAX,sentences,word2idx,tag2idx):
    X = [[word2idx[w[0]] for w in s] for s in sentences]
    X = pad_sequences(maxlen=MAX, sequences=X, padding="post",value=word2idx["<end>"])
    y = [[tag2idx[w[1]] for w in s] for s in sentences]
    y = pad_sequences(maxlen=MAX, sequences=y, padding="post", value=tag2idx["<e>"])
    y = to_categorical(y, num_classes=len(tag2idx))
    return X,y


def embeddings(embedding, word2idx):
    embeddings_index = dict()
    f = open(embedding)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    print('Loaded %s word vectors.' % len(embeddings_index))
    print("dimention is",len(coefs))
    embedding_matrix = np.zeros((len(word2idx), len(coefs)))
    for i,word in enumerate(word2idx.keys()):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector        
    
    return embedding_matrix

    
def evaluate_model(history):    
    plt.title('Accuracy')
    plt.plot(history.history['crf_viterbi_accuracy'], label='train')
    plt.plot(history.history['val_crf_viterbi_accuracy'], label='Val')
    plt.legend()
    plt.show();
  
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='Val')
    plt.legend()
    plt.show();
    
def pred2label(pred,idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(idx2tag[p_i])
        out.append(out_i)
    return out


## For Prediction
def Openfile(filename):
    Monolingual_sumerian=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            Monolingual_sumerian.append(line)
    return Monolingual_sumerian


def Savefile(Monolingual_sumerian,POS_list):
    with open(output, 'w') as f:
        for i in range(len(POS_list)):
            f.write("%s\n" %str(i+1))
            f.write("sentence: %s\n" %Monolingual_sumerian[i])
            f.write("NER:%s\n" % POS_list[i])
    print()

    
def preparedicts():

    with open("NER_Models/NER_Bi_LSTM_CRF/Sumerian_Vocab.pkl",'rb') as f:
    	vocabulary=pickle.load(f)
    word2idx,idx2word,tag2idx,idx2tag=vocabulary
    
    return word2idx,idx2word,tag2idx,idx2tag



def preparetestData(MAX,sentences,word2idx):
    X=[]
    for s in sentences:
        l=[]
        s=s.split()
        for w in s:
            try:
                l.append(word2idx[w])
            except KeyError:
                l.append(word2idx["UNK"])
        X.append(l)
    X = pad_sequences(maxlen=MAX, sequences=X, padding="post", value=word2idx["<end>"])
    return X

def pred2label(pred,idx2tag):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            tag=idx2tag[p_i]
            out_i.append(tag)
        out.append(out_i)
    return out


def Predict_Testtag(loaded_model,X,Monolingual_sumerian,idx2tag):
    test_pred = loaded_model.predict(X, verbose=1)
    y_pred = pred2label(test_pred,idx2tag)
    for i in range(len(Monolingual_sumerian)):
        s=Monolingual_sumerian[i].split()
        y_pred[i]=y_pred[i][:len(s)]
    return y_pred   



def POSLIST(Monolingual_sumerian,Prediction):
    my_list=[]
    for i in range(len(Monolingual_sumerian)):
        print(i+1)
        print("sentence: "+Monolingual_sumerian[i])
        l=Monolingual_sumerian[i].split()
        POS=""
        for j in range(len(l)):
            POS=POS+"("+l[j]+","+Prediction[i][j]+")"+" "
        print('NER:'+POS)
        my_list.append(POS)
        print()
    
    return my_list