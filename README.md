# Machine Learning Features

This code is intended to make using CRIPT for gathering polymer data for machine learning simple and easy.

Note: As of right now, only supervised learning support is being developed. Unsupervised learning support may be added in the future.

## How To Use: Returning Data Object

To return a data object, the user will call the **ML_Data_Process** function. This function has 5* parameters:

1. SearchProperties

These are the properties that the user wants to *search* for. For example, say a user is interested in searching for materials whose color property is red. This should be specified in this parameter

2. FeatureProperties

These are the properties the user wants to use as the features of their data. To continue the example from before, they may only want materials whose color property is red but want the properties that serve as the features for their model to be something such as molar mass, stress, solubility, etc. They should specify those properties here.

3. LabelProperties

These are the properties that serve as the labels for the model, similar to the features.

4. Preprocess

This parameter is where the user can specify if they want the function to preprocess the data for them. It can be set to False, in which case the function will do no preprocessing and only return a data object compatible with the user's machine learning library (if supported). If the user does want the function to preprocess the data for them, they can set Preprocess to a list of Boolean variables representing what types of preprocessing they want to do. As of right now, this list is of the form [Normalization, Outlier Removal, One Hot Encoding]. In other words, if the user is dealing with numeric data and wants it to be normalized and the outliers to be removed, they would set it to [True, True, False]. Or if they were dealing with categorical data and needed it to be one hot encoded, they would set it to [False, False, True].

5. Library

This is where the user specifies which of the supported machine learning libraries they are using. As of now, support is being developed for Scikit-learn, Tensorflow, and Pytorch. This is specified using an enum named ML_Library. For example, if the user is using pytorch, they would pass in the value ML_Library.PYTORCH

## How to Use: Exporting

There is another function in this code called **ML_Data_Save**. Instead of returning a data object compatible with a machine learning library, this saves the data into a csv file on the user's machine. This is useful if they wanted to instead analyze the data in R.

The use of this function is almost identical to the previous function. It has the same parameters except for Preprocess and Library. They function the same as in the previous function. The user calls this function with their parameters and it will save the material data into a csv file named "material_data"

NOTE: These features are still early in development, and as such do not currently function and use is heavily subject to change

