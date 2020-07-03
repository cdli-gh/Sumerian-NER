import numpy as np
import pandas as pd
import argparse
from collections import OrderedDict 
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF
from keras_contrib.losses import  crf_loss
from keras_contrib.metrics import crf_viterbi_accuracy



def Openfile(filename):
    Monolingual_sumerian=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            Monolingual_sumerian.append(line)
    return Monolingual_sumerian


def Savefile(Monolingual_sumerian,POS_list):
    with open(args.output, 'w') as f:
        for i in range(len(POS_list)):
            f.write("%s\n" %str(i+1))
            f.write("sentence: %s\n" %Monolingual_sumerian[i])
            f.write("POS:%s\n" % POS_list[i])
    print()

    
def preparedicts(df):
    vocab=list(df["FORM"].values)
    f = open(args.embedding)
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
    
    return word2idx,idx2word,tag2idx,idx2tag   
    


def preparetestData(sentences,word2idx):
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
            if tag=="<e>":
                tag="NE"
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
    for i in tqdm(range(len(Monolingual_sumerian))):
        print(i+1)
        print("sentence: "+Monolingual_sumerian[i])
        l=Monolingual_sumerian[i].split()
        POS=""
        for j in range(len(l)):
            if(re.search(r'\d+\(.+\)',l[j])):
                POS=POS+"("+l[j]+","+"NU"+")"+" "
            else:    
                POS=POS+"("+l[j]+","+Prediction[i][j]+")"+" "
        print('POS:'+POS)
        my_list.append(POS)
        print()
    
    return my_list



def main():

    Monolingual_sumerian=Openfile(args.input)
    loaded_model = load_model('Saved_Models/Bi_LSTM_CRF.h5',custom_objects={'CRF':CRF, 
                                                  'crf_loss':crf_loss, 
                                                  'crf_viterbi_accuracy':crf_viterbi_accuracy})
    df=pd.read_csv(Input_path)
    word2idx,idx2word,tag2idx,idx2tag= preparedicts(df)
    X=preparetestData(Monolingual_sumerian,word2idx)
    Prediction=Predict_Testtag(loaded_model,X,Monolingual_sumerian,idx2tag)
    POS_list=POSLIST(Monolingual_sumerian,Prediction)
    print("Saving_file "+args.output)
    Savefile(Monolingual_sumerian,POS_list)
    
    
    
    
    
    
    
    
if __name__=='__main__':
    # max sentence length is set to 50
    MAX=50
    Input_path='Dataset/Augmented_POSTAG_training_ml.csv'
    #Embedding_path='Word_Embeddings/sumerian_word2vec_50.txt'
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the Input text file to be  predicted", default="Dataset/sumerian_demo.txt")
    parser.add_argument("-s","--saved",help="Location of saved CRF weights in .h5 format", default="Saved_Models/POS/POS_Bi_LSTM_CRF.h5" )
    parser.add_argument("-o","--output",help="Location of output text file(Result)", default='Output/POS_Bi_LSTM_CRF.txt')
    parser.add_argument("-e","--embedding",help="Location of the embedding file", default="Word_Embeddings/sumerian_word2vec_50.txt")
    
    args=parser.parse_args()
    
    print("\n")
    print("Input file is ", args.input)
    print("Saved model is ", args.saved)
    print("Output file will be ", args.output)
    print("\n")
    
    main()
    

    
    
    
    
 
