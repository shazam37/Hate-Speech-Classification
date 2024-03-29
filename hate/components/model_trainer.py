import os
import sys
import pickle
import pandas as pd
from hate.logger import logging
from hate.constants import *
from hate.exception import CustomException
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences 
from hate.entity.config_entity import ModelTrainerConfig
from hate.entity.artifact_entity import ModelTrainerArtifacts, DataTransformationArtifacts
from hate.ml.model import ModelArchitecture


class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifacts: DataTransformationArtifacts):
        
        # self.model_trainer_config = model_trainer_config
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts


    def splitting_data(self, csv_path):
        try:
            logging.info("Entered the data splitting mode") 
            logging.info("Reading the data")
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into X and y")
            x = df[TWEET]
            y = df[LABEL]

            logging.info("Applying train test split on the data")
            X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
            print(len(X_train), len(y_train))
            print(len(X_test), len(y_test))
            print(type(X_train),type(y_train))
            logging.info("Exited the splitting data mode")
            return X_train, X_test, y_train, y_test
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def tokenizing(self,x_train):
        try:
            logging.info("Applying tokenization on the data")
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            logging.info(f"converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences,maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"The sequence matrix is: {sequences_matrix}")
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e   

    def initiate_model_trainer(self) -> ModelTrainerArtifacts:
        logging.info("Entered initiate model trainer mode")

        """
        Method Name :   initiate_model_trainer
        Description:    this function initiates the model trainer steps

        Output:         Returns model trainer artifact
        On Failure:     writes an exception log and raises exception
        """    

        try:
            logging.info("Entered the initiate model trainer function")
            print(f"The data transformation artifact lies here: {self.data_transformation_artifacts.transformed_data_path}")
            x_train, x_test, y_train, y_test = self.splitting_data(csv_path=self.data_transformation_artifacts.transformed_data_path)
            model_architecture = ModelArchitecture()

            model = model_architecture.get_model()

            logging.info(f"X_train size is: {x_train.shape}")

            logging.info(f"X_test size is: {x_test.shape}")

            sequences_matrix, tokenizer = self.tokenizing(x_train)

            logging.info("Entered into model training mode")

            model.fit(sequences_matrix, y_train,
                      batch_size=self.model_trainer_config.BATCH_SIZE,
                      epochs=self.model_trainer_config.EPOCH,
                      validation_split=self.model_trainer_config.VALIDATION_SPLIT)
            logging.info("Model training finished")

            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR, exist_ok=True)

            logging.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATH_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH
            )
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts
        
        except Exception as e:
            raise CustomException(e, sys) from e