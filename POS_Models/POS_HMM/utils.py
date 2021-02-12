import numpy as np
import pandas as pd
import re
from tqdm import tqdm

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
    for i in range(len(tagged_sentence)):
        tagged_sentence[i].insert(len(tagged_sentence[i]),(('<end>','<e>')))
    
    return tagged_sentence


def printing_details(tagged_sentence):
    print("\n")
    print('Example of tagged_sentence')
    print(tagged_sentence[2])
    print("\n")
    print('Dataset_information')
    print("Number of Tagged Sentences ",len(tagged_sentence))
    tagged_words=[tup for sent in tagged_sentence for tup in sent]
    print("Total Number of Tagged words", len(tagged_words))
    vocab=set([word for word,tag in tagged_words])
    print("Vocabulary of the Corpus",len(vocab))
    tags=set([tag for word,tag in tagged_words])
    print("Number of Tags in the Corpus ",len(tags))
    print("\n")
    
    
            
# compute Emission Probability
def word_given_tag(word, tag, train_bag):
    tag_list = [pair for pair in train_bag if pair[1]==tag]
    count_tag = len(tag_list)#total number of times the passed tag occurred in train_bag
    w_given_tag_list = [pair[0] for pair in tag_list if pair[0]==word]
    #now calculate the total number of times the passed word occurred as the passed tag.
    count_w_given_tag = len(w_given_tag_list)
    return (count_w_given_tag, count_tag)




# compute  Transition Probability
def t2_given_t1(t2, t1, train_bag):
    tags = [pair[1] for pair in train_bag]
    count_t1 = len([t for t in tags if t==t1])
    count_t2_t1 = 0
    for index in range(len(tags)-1):
        if tags[index]==t1 and tags[index+1] == t2:
            count_t2_t1 += 1
    return (count_t2_t1, count_t1)


def TransitionMatrix(tags, train_bag):
    tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
    for i, t1 in enumerate(list(tags)):
        for j, t2 in enumerate(list(tags)): 
            tags_matrix[i, j] = t2_given_t1(t2, t1, train_bag)[0]/t2_given_t1(t2, t1, train_bag)[1]    
    tags_df = pd.DataFrame(tags_matrix, columns = list(tags), index=list(tags))
    
    return tags_df

    

def rules_key(word):
    if(re.search(r'{d}',word) or re.search(r'{ki}',word) or re.search(r'lugal',word)):
        return 'NE'
    elif (re.search(r'.*\d+\(.*\)',word)):
        return 'NU'
    return 'N'
    

## For Prediction
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



def Openfile(filename):
    Monolingual_sumerian=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            Monolingual_sumerian.append(line)
    return Monolingual_sumerian


def Savefile(output,Monolingual_sumerian,POS_list):
    with open(output, 'w') as f:
        for i in range(len(POS_list)):
            f.write("%s\n" %str(i+1))
            f.write("sentence: %s\n" %Monolingual_sumerian[i])
            f.write("POS:%s\n\n" % POS_list[i])
    print()