import cript
import csv
from enum import Enum
import numpy as np
import sklearn.preprocessing

def ML_Data_Save(SearchProperties, FeatureProperties, LabelProperty):

    materials_list = cript.search(*Insert code to search for materials with Properties*)

    *Insert code that gets relevant information from materials_list*

    *Insert code that splits relevant information into training and test data sets*

    materials_no_outliers = 
    
    if FileFormat == "csv":
        
        train_file = open("training_data.csv", "a")
        test_file = open("test_data.csv", "a")

        train_writer = csv.writer(train_file)
        test_writer = csv.writer(test_file)

        train_writer.writerows(training data)
        test_writer.writerows(test data)

        train_file.close()
        test_file.close()

def sklearn_process(raw_data, preprocess):

    import sklearn

    new_data = [raw_data[0], raw_data[1]]

    if preprocess[0] == True:

        from sklearn.preprocessing import StandardScaler

        feature_standard = StandardScaler().fit_transform(new_data[0])

        new_data[0] = feature_standard

    if preprocess[1] == True:

        from sklearn.preprocessing import QuantileTransformer

        feature_quantile = QuantileTransformer(n_quantiles=round(0.2*len(new_data[0]))).fit_transform(new_data[0])

        new_data[0] = feature_quantile

    if preprocess[2] == True:

        from sklearn.preprocessing import OneHotEncoder

        feature_OneHot = OneHotEncoder(new_data[0])

    return new_data

def tensorflow_process(raw_data, preprocess):

    import tensorflow
    
    pass


def pytorch_process(raw_data, preprocess):

    import torch
    
    return None

class ML_Library(Enum):

    SKLEARN = 1
    TENSORFLOW = 2
    PYTORCH = 3

data_process_functions = {
    ML_Library.SKLEARN : sklearn_process,
    ML_Library.TENSORFLOW : tensorflow_process,
    ML_Library.PYTORCH : pytorch_process}

#Preprocess = [Normalization, Outlier Removal, One Hot Encoding]
def ML_Data_Process(SearchProperties, FeatureProperties, LabelProperty, Preprocess, Library):
    
    #SKLEARN uses array like objects for fit model

    #TENSORFLOW uses pandas.core.frame.DataFrame

    #PYTORCH uses torchvision.datasets.mnist.FasionMNIST

    materials_list = cript.Search(*Insert code to search for materials with Properties*)

    feature_matrix = []

    label_vector = []

    property_types = ["INSERT LIST OF PROPERTY TYPES"]

    for property in FeatureProperties:

        prop_list = []

        min = 10**10
        max = -(10**10)
        prop_type = None

        for material in materials_list:

            try:

                prop_list.append(material.property[property].value)

                label_vector.append(material.property[LabelProperty].value)

                #Might use this for normalization preprocessing step
                #Could cause complications with how the machine learning library user is using interprets the data

                #if material.property[property].type == float:
                    
                    #prop_type = float

                    #if material.property[property].value > max:

                        #max = material.property[property].value

                    #if material.property[property].value < min:

                        #min = material.property[property].value 

            except:
                print()
        
        #if Normalization == True and prop_type == float:

        #    for i in range(len(prop_list)):
        #        prop_list[i] = (prop_list[i] - min) / (max - min)

        feature_matrix.append(prop_list.copy())

    
    feature_matrix_np = np.array(feature_matrix)

    feature_matrix_np = np.transpose(feature_matrix_np)

    #Format for raw data = tuple with features as list of lists as first element, label values for second element
    return data_process_functions[Library]("regression", raw_data)

def ML_Data_Process2(ML_Type, Data, Library, Preprocess=None):
    
    #SKLEARN uses array like objects for fit model

    #TENSORFLOW uses pandas.core.frame.DataFrame

    #PYTORCH uses torchvision.datasets.mnist.FasionMNIST

    if Preprocess == None:

        if ML_Type.lower() == "regression":

            Preprocess = [False, True, False]

        elif ML_Type.lower() == "classification":

            Preprocess = [False, False, True]

    #Format for raw data = tuple with features as list of lists as first element, label values for second element
    return data_process_functions[Library](Data, Preprocess)





        