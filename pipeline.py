import re
import os
import subprocess
import unicodedata
import argparse
from shutil import rmtree

import importlib

stopping_chars=["@", "#", "&", "$"]



def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False
    
def OPEN(filename):
    lines=[]
    with open(filename, "r") as f:
        for line in f:
            line=line.strip()
            lines.append(line)

    return lines


def savefile(filename,LIST):
    with open(filename, 'w') as f:
        for line in LIST:
            f.write("%s\n" % line)




def processing_1(form):
    form=form.replace('#', '').replace('[', '').replace(']', '').replace('<', '').replace('>', '').replace( '!', '').replace('?', '').replace('@c','').replace('@t','').replace('_','').replace(',','')
    #x = re.sub(r"\[\.+\]","unk",text_line)
    #x = re.sub(r"...","unk",x)
    '''x = re.sub(r'\#', '', text_line)
    x = re.sub(r"\_", "", x)
    x = re.sub(r"\[", "", x)
    x = re.sub(r"\]", "", x)
    x = re.sub(r"\<", "", x)
    x = re.sub(r"\>", "", x)
    x = re.sub(r"\!", "", x)
    x = re.sub(r"@c", "", x)
    x = re.sub(r"@t", "", x)
    x = re.sub(r",", "", x)
    #x=re.sub(r"(x)+","x",x)
    x = re.sub(r"\?", "", x)
    x = x.split()
    x = " ".join(x)
    k = re.search(r"[a-wyzA-Z]+",x)
    if k:
        return x
    else:
        return "" '''
    if(len(form)==0):
        form="..." 
    return form 
                    
            
            
def Pipeline_start(lines):            
    Pipeline=[]
    for i,line in enumerate(lines):
        if len(line)>0 and is_number(line[0]):
            index=line.find(".")
            line=line[index+1:].strip()
            text=processing_1(line)
            Pipeline.append(text)
    return Pipeline
        
        
      


    
def Pipeline_end(lines):
    pipeline_result=[]
    
    #POS=OPEN(output_dir+'pos_pipeline.txt')
    #NER=OPEN(output_dir+'ner_pipeline.txt')
    tr_en=OPEN(output_dir+'trans_pipeline.txt')
    #pos_index=2;
    index=0
    for i,line in enumerate(lines):
        pipeline_result.append(line)
        if len(line)>0 and is_number(line[0]):
            #pipeline_result.append(POS[pos_index])
            #pipeline_result.append(NER[pos_index])
            pipeline_result.append('#tr.en:'+tr_en[index])
            index+=1
            #pos_index+=3
            
    return pipeline_result




def main():   
    if os.path.exists(output_dir):
    	rmtree(output_dir)
    os.makedirs(output_dir)
        
    lines=OPEN(input_path)
    if(atf_file=="False"):
    	Pipeline=[]
    	print(len(lines))
    	for line in lines:
    		Pipeline.append(processing_1(line))
    	savefile(output_dir+'pipeline.txt',Pipeline)
    else:
    	Pipeline=Pipeline_start(lines)
    	savefile(output_dir+'pipeline.txt',Pipeline)
    
    
    #POS MODEL
    print("\n Running Part of speech Model for Sumerian Language")
    if Flair==False:
        #os.system(f'python3 {pos_path} -i {output_dir}pipeline.txt -o {output_dir}pos_pipeline.txt')
        inp=output_dir+'/pipeline.txt'
        out = output_dir+'/pos_pipeline.txt'
        imp = importlib.import_module('POS_Models.{0}.{1}'.format(pos_tag, pos_tag))
        POS_tag = getattr(imp, '{0}'.format(pos_tag) )
        POS = POS_tag()
        POS.predict(input=inp, output=out)
    else:
        from FLAIR.predict import Predictions
        print("Using Flair Model")
        inp=output_dir+'/pipeline.txt'
        out=output_dir+'/pos_pipeline.txt'
        obj=Predictions(inp,out,True,False)
        obj.predict()
    
    #NER MODEL
    print("Running Named entity recognation Model for Sumerian Language")
    if Flair==False:
        #os.system(f'python3 {ner_path} -i {output_dir}pipeline.txt -o {output_dir}ner_pipeline.txt')

        # Dynamically call a class
        inp=output_dir+'/pipeline.txt'
        out = output_dir+'/ner_pipeline.txt'
        imp = importlib.import_module('NER_Models.{0}.{1}'.format(ner_tag, ner_tag))
        NER_tag = getattr(imp, '{0}'.format(ner_tag) )
        NER = NER_tag()
        NER.predict(input=inp, output=out)
    else:
        print("Using Flair Model")
        inp=output_dir+'/pipeline.txt'
        out=output_dir+'/ner_pipeline.txt'
        obj=Predictions(inp,out,False,True)
        obj.predict()
        
    #Translation MODEL
    print("Running Translation Model for Sumerian Language")
    model_name = trans_path.split('/')[-1].split('.')[0]
    
    if model_name == 'Transformer' or model_name == 'Back_Translation':
        if GPU==False:
            os.system(f'onmt_translate -model {trans_path} -src {output_dir}pipeline.txt -output {output_dir}trans_pipeline.txt -replace_unk -verbose')
        else:
    	    os.system(f'CUDA_VISIBLE_DEVICES=0 onmt_translate -model {trans_path} -src {output_dir}pipeline.txt -output {output_dir}trans_pipeline.txt -replace_unk -verbose -gpu 0')
    if model_name == 'XLM' or model_name == 'MASS':
        if GPU==False:
    	    os.system(f'sh Translation_Models/evalXLM.sh ../{output_dir}pipeline.txt {model_name} ../{output_dir}trans_pipeline.txt')
        else:
    	    os.system(f'CUDA_VISIBLE_DEVICES=0 sh Translation_Models/evalXLM.sh ../{output_dir}pipeline.txt {model_name} ../{output_dir}trans_pipeline.txt')
    
    #Converting POS_NER to conll
    
    if(atf_file!="False"):
    	print("\n converting POS_NER to conll form")
    	os.system(f'python ATF_2_Conll/atf2conll_tags.py -i {input_path}')
    
        
    if(atf_file=="False"):
    	print("\n Generating atf file output")
    	pipeline_result=Pipeline_end(lines)
    	savefile(output_dir+'pipeline_output.atf',pipeline_result)
    
    print("\n Done!.. \n")
    
if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-i","--input",help="Location of the Input ATF File ", default="ATF_INPUT/demo.atf")
    parser.add_argument("-p","--pos",help="POS Model to be used from ['POS_CRF','POS_HMM','POS_Bi_LSTM','POS_Bi_LSTM_CRF'] (Case sensitive)", choices=['POS_CRF','POS_HMM','POS_Bi_LSTM','POS_Bi_LSTM_CRF'],default="POS_CRF" )
    parser.add_argument("-n","--ner",help="NER Model to be used from ['NER_CRF','NER_Bi_LSTM','NER_Bi_LSTM_CRF'] (Case_sensitive)", choices=['NER_CRF','NER_Bi_LSTM','NER_Bi_LSTM_CRF'],default="NER_CRF" )
    parser.add_argument("-t","--trans",help="Machine Translation Model to be used",choices=['Transformer', 'Back_Translation', 'XLM', 'MASS'], default="Back_Translation" )
    parser.add_argument("-o","--output",help="Location of output Directory", default='ATF_OUTPUT/')
    parser.add_argument("-a","--atf",help="Use of Text file rather than ATF", default=True)
    parser.add_argument("-g","--gpu",help="Use of GPU if avaliable", default=False)
    parser.add_argument("-f","--flair",help="Use of flair language model", default=False)

    
    args=parser.parse_args()
    #check
    ner_tag = args.ner
    pos_tag = args.pos

    input_path=args.input
    pos_path='POS_Models/'+args.pos+'/prediction.py'
    ner_path='NER_Models/'+args.ner+'/prediction.py'
    trans_path='Translation_Models/'+args.trans+'.pt'
    output_dir=args.output
    atf_file=args.atf
    GPU=args.gpu
    Flair=args.flair
    
    print("\n")
    print("Input file is ", input_path)
    print("POS Model path is ", pos_path)
    print("NER Model path is ", ner_path)
    print("Translation Model path is ", trans_path)
    print("Output directory is", output_dir)
    print("Using ATF file", atf_file)
    print("GPU", GPU)
    print("Flair_Model", Flair)
    print("\n")
    
    main()
    
    
    
    
    
    
    #IF WE WANT TO USE CLASS 
    #pipeline=OPEN('ATF_OUTPUT/pipeline.txt')
    #POS=POS_tag(Pipeline)
    #savefile('ATF_OUTPUT/pipeline1.txt',POS)
    #NER=NER_tag(Pipeline)
    #savefile('ATF_OUTPUT/pipeline2.txt',NER)
    #translations=Translation_tag(Pipeline)
    #savefile('ATF_OUTPUT/pipeline3.txt',translations)
