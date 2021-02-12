from .NER_CRF_features import features
from tqdm import tqdm

## From training.py
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
    
    
            
def word_list(sentence):
    list_of_words=[]
    for word,tag in sentence:
        list_of_words.append(word)
    return list_of_words


def prepareData(tagged_sentences):
    X,y=[],[]
    for index,sentence in enumerate(tagged_sentences):
        single_sentence_feature=[]
        # Preparing features of all words of a single sentence/phrase
        for i in range(len(sentence)):
            #word list of sentence
            list_of_words=word_list(sentence)
            #feature of word at index i
            d=features(list_of_words,i)
            single_sentence_feature.append(d)
            
        X.append(single_sentence_feature)
        # append list of tags for the associated sentence
        y.append([tag for word,tag in sentence])
    return X,y

##From Prediction.py
def Openfile(filename):
    Monolingual_sumerian=[]
    with open(filename) as f:
        for line in f:
            line=line.strip()
            Monolingual_sumerian.append(line)
    return Monolingual_sumerian


def test_word_list(sentence):
    list_of_words=sentence.split()
    return list_of_words


def prepare_test_Data(all_sentences):
    X=[]
    for sentence in all_sentences:
        single_sentence_feature=[]
        #word list of sentence
        list_of_words=test_word_list(sentence)
        # Preparing features of all words of a single sentence/phrase
        for i in range(len(sentence.split())):
            #feature of word at index i
            d=features(list_of_words,i)
            single_sentence_feature.append(d)

        X.append(single_sentence_feature)  
    return X


def POSLIST(Monolingual_sumerian,Prediction):
    my_list=[]
    for i in tqdm(range(len(Monolingual_sumerian))):
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


def Savefile(output, Monolingual_sumerian,POS_list):
    with open(output, 'w') as f:
        for i in range(len(POS_list)):
            f.write("%s\n" %str(i+1))
            f.write("sentence: %s\n" %Monolingual_sumerian[i])
            f.write("NER:%s\n" % POS_list[i])
    print()

