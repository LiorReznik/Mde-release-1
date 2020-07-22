# -*- coding: utf-8 -*-
"""
@author: Lior Reznik
The work is licensed under the MIT licence, for more details: 
    https://github.com/LiorReznik/Mde-release-1/blob/master/LICENSE
"""
from Data import DataPreparation
from argparse import ArgumentParser
from logger import Logger
from predict import Model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

def score_func(**kwargs)->None:
    def read_data():
        logger.info("Reading data")
        with open(kwargs.get('data_path'),"r",encoding="utf8", errors='ignore') as f:
            return f.read().strip().split("\n")
        
    def write_res():
       logger.info("writing results")
       with open(kwargs.get("save_path","./data_annoated.txt"),"w",encoding="utf8") as f:
           f.writelines("{}: {}\n".format(L[0],L[1]) for L in pred)
           
    with Logger("Log") as logger:
        data = read_data()
        prep =DataPreparation(logger)
        model_type = kwargs.get("model_type","cnn_mld")
        preprocessed_data=prep(data=data,depth=model_type.split("_")[-1])
        pred = Model(logger=logger)(model_type=model_type,data=preprocessed_data)
        pred = zip(list(map(lambda x:{1:"Def",0:"Not Def"}[x],pred)),data)
        write_res()

if __name__ == "__main__":
    def check_input():
        if not data_path:
            parser.error("you must enter input_path")
        if not data_path.endswith(".txt"):
            parser.error("you must enter input_path to a txt file")
        if  os.stat(data_path).st_size<=0:
            parser.error("the txt file is empty")
        if model_type not in {None,"cblstm_ml","cblstm_mld","cnn_mld"}:
            parser.error("you must enter valid model name")
       
        if not {'Data.py','logger.py','main.py','predict.py',
                'singleton.py',}.issubset(os.listdir('.')):
           parser.error("""you project dir is incomplete,
                        the project's main dir should conatin:
                            'Data.py','logger.py','main.py',
                            'predict.py','singleton.py""") 
                            
        models = {'wolfram_cblstm_mld_FastText.model','wolfram_cblstm_ml_FastText.model',
                 'wolfram_cnn_mld_FastText.model','wolfram_deps2ids.pkl',
                 'wolfram_ids2deps.pkl','wolfram_maxlen.pkl'}
        
        if (os.listdir('./models') and not  models.issubset(os.listdir('./models'))) and not models.issubset(os.listdir('.')):
            parser.error("""your main folder should conatin models folder with the following files:
                            'wolfram_cblstm_mld_FastText.model',
                            'wolfram_cblstm_ml_FastText.model',
                            'wolfram_cnn_mld_FastText.model',
                            'wolfram_deps2ids.pkl',
                            'wolfram_ids2deps.pkl',
                            'wolfram_maxlen.pkl'""") 
        


    parser = ArgumentParser(description='MDE!.your math Ranker!')
    parser.add_argument('-i','--input_path',type=str,
                    help="""enter full path to an txt file that you want to annotate""")
    parser.add_argument('-o','--output_path',type=str,
                        help="""optional:enter full path to save the results in""")
    
    parser.add_argument('-m','--model',type=str,
                        help="""optional:enter the desired model
                                your options are:
                                    cblstm_ml
                                    cblstm_mld
                                    cnn_mld (Default)
			""")

    args = vars(parser.parse_args())
    data_path,model_type,save_path=args.get("input_path"),args.get("model"),args.get("output_path")
    check_input()
    args = {"data_path":data_path}
    if save_path:
        args["save_path"] = save_path
    if  model_type:
        args["model_type"] = model_type
    score_func(**args) 

         