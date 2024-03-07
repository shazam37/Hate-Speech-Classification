from dataclasses import dataclass

# Data-ingestion-artifacts

@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str 

# Model-trainer-artifacts

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    x_test_path: str
    y_test_path: str

# Data-transformation-artifacts

# @dataclass
# class DataTransformationArtifacts:
#     def __init__(self, transformed_data_path):
#         self.transformed_data_path: str = transformed_data_path

#     def return_transformed_data_path(self):
#         return self.transformed_data_path

@dataclass
class DataTransformationArtifacts:
    transformed_data_path: str

# Model-evaluation-artifacts

@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool

# Model-pusher-artifacts

@dataclass
class ModelPusherArtifacts:
    bucket_name: str