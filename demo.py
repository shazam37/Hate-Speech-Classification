from hate.logger import logging
from hate.exception import CustomException
import sys

from hate.configuration.gcloud_syncer import GCloudSync

logging.info("Welcome to the project")

# try:
#     a = 7 // "0"
# except Exception as e:
#     raise CustomException(e,sys) from e

obj = GCloudSync()
obj.sync_folder_from_gcloud('hate_speech_classification', 'dataset.zip', 'download/dataset.zip')