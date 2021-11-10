from azureml.core import Workspace
from azureml.core.webservice import LocalWebservice
from azureml.core.model import InferenceConfig, Model
from random import randint

ws = Workspace.from_config()

model = Model(workspace=ws, name="test-triton-model")

service_name = "triton-bidaf-9" + str(randint(10000, 99999))

config = LocalWebservice.deploy_configuration(port=6789)

service = Model.deploy(
    workspace=ws,
    name=service_name,
    models=[model],
    deployment_config=config,
    overwrite=True,
)

service.wait_for_deployment(show_output=True)
