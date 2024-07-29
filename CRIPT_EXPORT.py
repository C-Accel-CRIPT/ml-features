from cript import *
from enum import Enum
import numpy as np
from dotenv import load_dotenv

""" def ML_Data_Save(SearchProperties, FeatureProperties, LabelProperty):

    import csv

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
        test_file.close() """

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

        new_data[0] = feature_OneHot

    return new_data

def tensorflow_process(raw_data, preprocess):

    import tensorflow
    
    pass

def pytorch_process(raw_data, preprocess):

    import torch

    from torch.utils.data import Dataset

    if preprocess != False and preprocess != [False, False, False]:

        from torchvision.transforms import v2

    tensor_X = torch.tensor(raw_data[0])

    tensor_y = torch.tensor(raw_data[1])

    class MaterialDataset(Dataset):

        def __init__(self, X, y, transform, transform_target):

            self.labels = y
            self.features = X
            self.transform = transform
            self.transform_target = transform_target
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            
            feature = self.features[idx]
            label = self.features[idx]

            if self.transform:
                feature = self.transform(feature)

            if self.transform_target:
                feature = self.transform(label)
            
            return feature, label

    """     if type(preprocess[0]) == list:

        preprocess[0] = v2.Compose(preprocess[0])

    elif preprocess[0] == False:
        preprocess[0] = None

    if type(preprocess[1]) == list:

        preprocess[1] = v2.Compose(preprocess[1])

    elif preprocess[1] == False:
        preprocess[1] = None """
        
    #mat_dataset = MaterialDataset(tensor_X, tensor_y, preprocess[0], preprocess[1])

    mat_dataset = MaterialDataset(tensor_X, tensor_y, None, None)
    
    return mat_dataset

class ML_Library(Enum):

    SKLEARN = 1
    TENSORFLOW = 2
    PYTORCH = 3

data_process_functions = {
    ML_Library.SKLEARN : sklearn_process,
    ML_Library.TENSORFLOW : tensorflow_process,
    ML_Library.PYTORCH : pytorch_process}

#Preprocess = [Normalization, Outlier Removal, One Hot Encoding]

def Mat_Search(Properties, SearchParameter, limit, getname = False):
    
    load_dotenv()

    result_list = Search(node="Material", q=SearchParameter, filters={"limit": limit})

    get_smile = False

    if "smiles" in Properties:
        get_smile = True

    if "name" in Properties:
        getname = True

    material_list = []

    for material in result_list:

        mat_dic = {}

        if getname == True:

            mat_dic["name"] = material["name"]
        
        if get_smile == True:
            
            mat_dic["smiles"] = material["smiles"]

        for property in material["property"]:

            if (property["key"] in Properties) and property['value'] != '':
                
                try:
                    mat_dic[property["key"]] = property["value"]
                
                except:
                    pass

        add = True

        if len(mat_dic) > 0:
            
            for prop in Properties:

                if prop not in mat_dic.keys():
                    add = False
                    break

            if add == True:
                material_list.append(mat_dic.copy())

    return material_list

def ML_Data_Save(Properties, Limit=10000, SearchParameter = ''):

    import csv

    load_dotenv()

    materials_list = Mat_Search(Properties, SearchParameter, Limit, getname=True)

    mat_matrix = [list(materials_list[0].keys())]

    for mat in materials_list:

        mat_matrix.append(list(mat.values()))

    data_file = open(f"material_data_{SearchParameter}_{Limit}.csv", "a", newline='')

    data_writer = csv.writer(data_file)

    data_writer.writerows(mat_matrix)

    data_file.close()

def ML_Data_Process(FeatureProperties, LabelProperties, Library, SearchParameter='', Preprocess=False, limit=10000):
    
    #SKLEARN uses array like objects for fit model

    #TENSORFLOW uses pandas.core.frame.DataFrame

    #PYTORCH uses torchvision.datasets.mnist.FasionMNIST

    load_dotenv()

    properties = FeatureProperties + LabelProperties

    material_list = Mat_Search(properties, SearchParameter, limit)

    feature_list = []
    label_list = []

    for material in material_list:

        mat_feat_list = []
        mat_label_list = []

        for prop in FeatureProperties:
            
            mat_feat_list.append(material[prop])

        for prop in LabelProperties:
            
            mat_label_list.append(material[prop])

        feature_list.append(mat_feat_list.copy())
        label_list.append(mat_label_list.copy())

    if Preprocess == False:

        Preprocess = [False, False, False]

    data = (feature_list, label_list)

    #Format for raw data = tuple with features as list of lists as first element, label values for second element
    return data_process_functions[Library](data, Preprocess)

""" def ML_Data_Process2(SearchProperties, FeatureProperties, LabelProperty, Preprocess, Library):
    
    #SKLEARN uses array like objects for fit model

    #TENSORFLOW uses pandas.core.frame.DataFrame

    #PYTORCH uses torchvision.datasets.mnist.FasionMNIST

    #materials_list = cript.Search(*Insert code to search for materials with Properties*)

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
    return data_process_functions[Library]("regression", raw_data) """






        
