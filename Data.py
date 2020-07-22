# -*- coding: utf-8 -*-
"""
@author: Lior Reznik
The work is licensed under the MIT licence, for more details: 
    https://github.com/LiorReznik/Mde-release-1/blob/master/LICENSE
"""
import numpy as np
import os, pickle
from collections import OrderedDict
import urllib,zipfile          
from stanza.server import CoreNLPClient
from singleton import Singleton

class DataPreparation(metaclass=Singleton):
    def __init__(self,logger):
        self.logger = logger
        self.doanload_manager  
        self.name = "wolfram"

    def __call__(self,data:list,depth:str):
        """
        method to manage all the preprocessing pipe

        Parameters
        ----------
        data : list
            raw data to be preprocessed. 
        depth : str
            the depth of preprocessing.

        Returns
        -------
        numpy
            preprocessed data.

        """
        print(depth)
        self.__instances = data
        # init the preprocessed vocab
        self.preprocess = {"X": [], "M": [], "ML": [], "MLD": [], "NONE": [],
                           "depth": depth.lower(), "maxlen": 0,'deps2ids':  OrderedDict()}
        self.preprocessing_data
        return self.preprocess['X']

    @property            
    def doanload_manager(self):
        """
        method to manage the download and extraction of
        stanfordcoreNLP server and fastText vectors

        Returns
        -------
        None.

        """
        if not os.path.exists("./data"):
            os.makedirs("./data")
        
        for path in (( "http://nlp.stanford.edu/software/stanford-corenlp-4.0.0.zip",
                       "./data/corenlp.zip","./data","stanford-corenlp-4.0.0",True),
                      ( "http://nlp.stanford.edu/software/stanford-corenlp-4.0.0-models-english.jar",
                       "./data/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0-models-english.jar","./data/stanford-corenlp-4.0.0","stanford-corenlp-4.0.0-models-english.jar",False),
                      ( "http://nlp.stanford.edu/software/stanford-corenlp-4.0.0-models-english-kbp.jar",
                       "./data/stanford-corenlp-4.0.0/stanford-corenlp-4.0.0-models-english-kbp.jar","./data/stanford-corenlp-4.0.0","stanford-corenlp-4.0.0-models-english-kbp.jar",False),
                      ( "https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip",
                       "./data/fastText.zip","./data","wiki-news-300d-1M.vec",True),  
                     ):
                       self.download_resorces(download_path=path[0],
                                              extract_archive=path[1],
                                              extract_path=path[2],
                                              extraction_name=path[-2],
                                              extract=path[-1]  
                                              )        

        os.environ["CORENLP_HOME"] = "./data/stanford-corenlp-4.0.0"
            
    def download_resorces(self,**kwargs):
        """
        download and extract resorces from the net

        Returns
        -------
        None.

        """
        def download():
            try:
                self.logger.info("Downloading: {}".format(kwargs.get("download_path")))
                _, _ = urllib.request.urlretrieve(kwargs.get("download_path"), kwargs.get("extract_archive"))
                self.logger.info("Download has completed")
            except urllib.error as e:
                self.logger.info(e.to_str()) 
                
        def extract():
            self.logger.info("starting the extraction of {}".format(kwargs.get("extract_archive")))
            with zipfile.ZipFile(kwargs.get("extract_archive"), 'r') as zip_ref:
                 zip_ref.extractall(kwargs.get("extract_path","./data"))
            self.logger.info("done extracting")
            
        if not os.path.exists(kwargs.get("extract_archive")):
            download()

        else:
            self.logger.info("Skipping Download,The Archive already in the HD")
        if kwargs.get("extract",True) and not os.path.exists("{}/{}".format(kwargs.get("extract_path"),kwargs.get("extraction_name"))):
            extract()
        else:
            self.logger.info("Skipping extraction,The folder already in the HD")

    @property
    def load_embeddings(self) -> tuple:
        """
        Method to load fastText vectors

        Returns
        -------
        tuple
            vocab,word vectors and dims of the vectors.

        """
        self.logger.info("starting to load embeddings")
        with open(os.path.join(os.getcwd(),"./data/wiki-news-300d-1M.vec"), 'r', encoding='utf-8', newline='\n', errors='ignore') as f:
            _, dims = map(int, f.readline().split())
            model = {}
            for line in f:
                tokens = line.rstrip().split(' ')
                model[tokens[0]] = np.asarray(tokens[1:], "float32")
            vocab =set(model.keys())
        self.logger.info("Done Loading Emmbedings")
        return vocab, model, dims

    @property
    def preprocessing_data(self):
        """
        Method that takes all the instances in the data and preprocess them.
        """

        def pad_words(tokens:list, append_tuple:bool=False)->list:
            """""
            Function to pad/shrink the sentence to the maxlen length
            """""
            #   shrinking
            if len(tokens) > self.preprocess['maxlen']:
                return tokens[:self.preprocess['maxlen']]
            else:
                #   padding
                for _ in range(self.preprocess['maxlen'] - len(tokens)):
                    tokens.append(('UNK', 'UNK') if append_tuple else 'UNK')
                return tokens

        def prep_func():
            def final_tuning():
                """""
                function to add all of the matrices together
                """""
                del self.__instances
                self.logger.info("in final tuning")
                if self.preprocess['depth'] == 'm':
                    del self.preprocess['ML']
                    del self.preprocess['MLD']
                    self.preprocess['X'] = np.concatenate([self.preprocess["NONE"],
                                                           self.preprocess['M']], axis=1)
                    print(self.preprocess['X'])
                elif self.preprocess['depth'] == 'ml':
                    del self.preprocess['M']
                    del self.preprocess['MLD']
                    self.preprocess['X'] = np.concatenate([self.preprocess["NONE"],
                                                           self.preprocess["ML"]], axis=1)
                else:
                    del self.preprocess['M']
                    del self.preprocess['ML']
                    self.preprocess['X'] = np.array(self.preprocess["MLD"])

            def get_depth():
                """""
                function to find the depth of all the words in the dependency tree
                """""
                nonlocal tree
                number = 0
                while True:
                    keys = [key for key, value in tree.items() if value == number]
                    if not keys:
                        break
                    number += 1
                    tree = {key: number if value in keys else value for key, value in tree.items()}
           
            def get_pairs_and_dep():
                """""
                function that parses a json_obj that is a result of dependency parsing from stanford core nlp server
                :return dict of dicts that contains  pair and dep ->word pairs(parent,child) and dependencies tree
                """""
                nonlocal pairs_dep, tree

                stanford_obj = self.__nlp.annotate(sent, properties={'annotators': "depparse",
                                                                               'outputFormat': 'json', 'timeout':'5000000'})
                   
                
             
                pairs_dep, tree = OrderedDict(), {}
                tree[(stanford_obj['sentences'][0]['basicDependencies'][0]['dependentGloss'],
                      stanford_obj['sentences'][0]['basicDependencies'][0]['dependent'])] = 0

                for index, dict in enumerate(stanford_obj['sentences'][0]['basicDependencies'][1:]):
                    tree[(dict['dependentGloss'], dict['dependent'])] = (dict['governorGloss'], dict['governor'])
                    pairs_dep[index] = {
                        'word_pair': (dict['dependentGloss'], dict['governorGloss'], dict['dependent']),
                        'dependency': dict['dep'],
                    }


            def build_lists_of_pairs():
                """""
                function that builds and returns tuple of lists : (1ist of word pairs, list of dependencies)
                """""
                nonlocal word_pairs, dependencies
                words_pairs, dependencies = [], []
                for token in pairs_dep.values():
                    words_pairs.append((token['word_pair']))
                    dependencies.append(token['dependency'])
                word_pairs, dependencies = pad_words(words_pairs, append_tuple=True), pad_words(dependencies)

            def build_head_modifier_vectors():
                """""
                function that builds a vector out of the  (head,modifier) pair in case of m/ml avg vector of 300 dims 
                in case of mld vector [head,modifier] of 600 dims
                """""
                nonlocal head_modifer_vec
                
                head, modifier = word_pair[0], word_pair[1]
                head_vec = model[head] if head != 'UNK' and head in vocab else np.zeros(dims)
                modifier_vec = model[modifier] if modifier != 'UNK' and modifier in vocab else np.zeros(dims)

                head_modifer_vec = np.concatenate([head_vec, modifier_vec]) if self.preprocess['depth'] == 'mld'\
                    else np.mean(np.array([head_vec, modifier_vec]), axis=0)

            def build_dependency_vector():
                """""
                building one-hot encodding vector for the dependency labels
                """""
                nonlocal dependency_vector
                # initializing a vector for the dependency with 0's
                dependency_vector = np.zeros(46)

                # pouting one in the index of the label \if the label is UNK then pouting one in the end
                dependencies_index = -1 if dependency_labels[ind] == 'UNK' or dependency_labels[ind] not in \
                                           self.preprocess['deps2ids'].keys() else \
                    self.preprocess['deps2ids'][dependency_labels[ind]]
                dependency_vector[dependencies_index] = 1

            head_modifer_vec, dependency_vector, depths, pairs_dep, tree, word_pairs, \
            dependencies = [], [], {}, {}, {}, [], []
            l =len(self.__instances)
            for idx, sent in enumerate(self.__instances):
                if idx != 0 and idx % 10 == 0:
                    self.logger.info('{} prec of pairs is done'.format(idx/l * 100))   
 
                get_pairs_and_dep()
                build_lists_of_pairs()
                dependency_labels = [i for i in dependencies]
                words_pair_label_depth = []
                avg_sent_matrix = []
                avg_label_sent_matrix = []
                if self.preprocess['depth'] == 'mld':
                    get_depth()

                for ind, word_pair in enumerate(word_pairs):
                    build_head_modifier_vectors()
                    # m ->for each(H,M) building avg(H,M)-> 300 vec+ 46 zeros(to fit the matrix)
                    if self.preprocess['depth'] == 'm':
                        avg_sent_matrix.append(np.concatenate([head_modifer_vec, np.zeros(46)]))
                    else:
                        build_dependency_vector()
                        vecs = np.concatenate([head_modifer_vec, dependency_vector])
                        if self.preprocess['depth'] == 'ml':
                            # ml ->for each(H,M) building avg(H,M)+L-> 346 vec
                            avg_label_sent_matrix.append(vecs)
                        else:
                            # mld ->for each(H,M) building H,M,L,Depth of H-> 647 vec
                            vecs = np.concatenate([vecs, np.array(
                                [tree[(word_pair[0], word_pair[-1])] if (word_pair[0], word_pair[-1]) in tree.keys()
                                 else -1])])
                            words_pair_label_depth.append(vecs)

                self.preprocess['MLD'].append(np.array(words_pair_label_depth))
                del words_pair_label_depth

                self.preprocess['M'].append(np.array(avg_sent_matrix))
                del avg_sent_matrix

                self.preprocess['ML'].append(np.array(avg_label_sent_matrix))
                del avg_label_sent_matrix
            
            final_tuning()

        def load():
            def load_obj(name):
                with open(name + '.pkl', 'rb') as f:
                    return pickle.load(f)
            self.logger.info("loading deps")
            def manage(path):
                self.preprocess['ids2deps'] = load_obj("{}/{}_ids2deps".format(path,self.name ))
                self.preprocess['deps2ids'] = load_obj("{}/{}_deps2ids".format(path,self.name ))
                self.preprocess['maxlen'] = load_obj("{}/{}_maxlen".format(path,self.name ))
            try:
                manage("models") 
            except OSError:
                manage(".")
            
            self.logger.info("done loading deps")

        def preprocess_none():
            """""
            function to preprocess none level
            """""
            def tokenize_sent():
                """""
                function to tokenize a sentence
                """""
                nonlocal tokens

                data=self.__nlp.annotate(sent, properties={'annotators': 'tokenize', 'outputFormat': 'json'})
                tokens = [token['word'].lower() for token in data['tokens']]

            def sent_to_matrix():
                """""
                building a  matrix out of the sentences, each vector in the matrix will be at the same length
                """""
                matrix = []
    
                for token in pad_words(tokens, self.preprocess['maxlen']):

                    matrix.append(np.concatenate([model[token], np.zeros(46)])
                      if token in vocab else np.zeros(dims + 46))

                self.preprocess['NONE'].append(np.array(matrix))
                del matrix

            tokens = []
            l = len(self.__instances)
            for idx, sent in enumerate(self.__instances):
                if idx % 50 == 0:
                    self.logger.info('done {} prec of the sentence matrix building'.format(idx/l*100))

                    
                tokenize_sent()
                sent_to_matrix()
            self.preprocess['NONE'] = np.array(self.preprocess['NONE'])

        self.logger.info("start nlp server")
        with CoreNLPClient(memory='4G', endpoint='http://localhost:9001') as self.__nlp:
                self.logger.info("opened nlp")  
                vocab, model, dims = self.load_embeddings
                load() 
                if self.preprocess['depth'] != 'mld':
                          preprocess_none()
              
                print("prep func")
                prep_func()
                self.logger.info("prep is done with shape:{}".format(self.preprocess['X'].shape))

            