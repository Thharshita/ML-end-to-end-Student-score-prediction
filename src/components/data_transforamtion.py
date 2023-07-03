import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocesor_obj_file_path= os.path.join("artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config= DataTransformationConfig()

    def get_data_transformer_object(self):

        """
        This function is responsible for data_transformation"""

        try:
            numerical_columns=["reading_score","writing_score"]
            categorical_columns=["gender","race_ethnicity","parental_level_of_education","lunch","test_preparation_course"]


            num_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler(with_mean=False, with_std= False))
                ]
            )


            cat_pipeline= Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False, with_std= False))
                ]
            )

            logging.info(f"Numeriacal columns:{numerical_columns}")

            logging.info(f"Catogorical columns:{categorical_columns}")


            preprocessor= ColumnTransformer(
                [
                    ("num_pipelines", num_pipeline, numerical_columns),
                    ("cat_pipelines", cat_pipeline, categorical_columns)
                ]
            )
            #The preprocessor object contains the information about which columns in the input dataset
            #  should undergo specific transformations. In this case, it has the names of the numerical
            #  columns and categorical columns that will be transformed during the training phase.


            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)


    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df= pd.read_csv(train_path)
            test_df= pd.read_csv(test_path)
            logging.info("Read train test data done")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj= self.get_data_transformer_object()

            target_column_name= "math_score"
            numericals_name= ["reading_score","writing_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df=test_df[target_column_name]


            logging.info("Applying preprocessing object on trainig dataframe and testing dataframe")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)

            train_arr=np.c_[input_feature_train_arr,np.array(input_feature_train_arr)]
            test_arr=np.c_[input_feature_test_arr,np.array(input_feature_test_arr)]

            save_obj(
                file_path=self.data_transformation_config.preprocesor_obj_file_path,
                obj= preprocessor_obj
            )

            logging.info("Saved preprocessing obj")


            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocesor_obj_file_path
            )

            

            


        except Exception as e:
            raise CustomException(e,sys)