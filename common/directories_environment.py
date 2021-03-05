# %% --------------------
import os
import sys

from dotenv import load_dotenv

# local
# env_file = "D:/GWU/4 Spring 2021/6501 Capstone/VBD CXR/PyCharm " \
#           "Workspace/vbd_cxr/6_environment_files/local.env "
# cerberus
env_file = "/home/ssebastian94/vbd_cxr/6_environment_files/cerberus.env"

load_dotenv(env_file)

# add HOME DIR to PYTHONPATH
sys.path.append(os.getenv("HOME_DIR"))

# DIRECTORIES
SAVED_MODEL_PATH = os.getenv("SAVED_MODEL_DIR") + "/saved_model_20210212.pt"
VALIDATION_INDICES = os.getenv("VALIDATION_INDICES")
IMAGE_DIR = os.getenv("IMAGE_DIR")
BB_FILE = os.getenv("BB_FILE")
VALIDATION_PREDICTION_DIR = os.getenv("VALIDATION_PREDICTION_DIR")
