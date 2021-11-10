from azureml.core.model import Model
from azureml.core import Workspace

model_path = "/Users/luca/Projects/triton-inference-server/models"

ws = Workspace.from_config()

model = Model.register(
        model_path=model_path,
        model_name="test-triton-model",
        model_framework=Model.Framework.MULTI,
        workspace=ws
        )
