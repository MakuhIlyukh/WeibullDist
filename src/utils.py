import os
import shutil

import mlflow
from mlflow.utils.git_utils import get_git_commit
from mlflow.utils.mlflow_tags import MLFLOW_GIT_COMMIT


def set_commit_tag():
    """ Sets commit tag under the current run.
    
    By default, mlflow doesn't set commit tag in interactive mode
    (jupyter notebooks or ipython session)
    """
    comm_hash = get_git_commit(".")
    mlflow.set_tag(MLFLOW_GIT_COMMIT, comm_hash)


def del_folder_content(folder):
    """ Deletes the contents of a folder. """
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print('Failed to delete %s. Reason: %s' % (file_path, e))