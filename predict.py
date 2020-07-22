# -*- coding: utf-8 -*-
"""
@author: Lior Reznik
The work is licensed under the MIT licence, for more details: 
    https://github.com/LiorReznik/Mde-release-1/blob/master/LICENSE
"""
from keras.models import load_model
import numpy as np
from singleton import Singleton
class Model(metaclass=Singleton):
    def __init__(self,logger):
        self.logger = logger
    def __call__(self,data:list,model_type:str):
        self.logger.info("loading model")
        try:
            self.model = load_model("models/wolfram_{}_FastText.model".format(model_type))   
        except OSError:
            self.model = load_model("./wolfram_{}_FastText.model".format(model_type))   

        self.logger.info("predicting")
        return np.array([i[0] for i in self.model.predict_classes(data)])
     
       