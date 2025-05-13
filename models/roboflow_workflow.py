import os

from inference_sdk import InferenceHTTPClient

from .model import Model


class RoboflowWorkflow(Model):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = InferenceHTTPClient(
            api_url="https://detect.roboflow.com",
            api_key=os.environ.get("ROBOFLOW_API_KEY"),
        )

    def run(
        self,
        image: str,
        prompt: str,
        image_name: str = "image.png",
        structured_output_format=None,
    ):
        return self.client.run_workflow(
            workspace_name="capjamesg",
            workflow_id="custom-workflow-18",
            images={
                "image": image_name,
            },
            parameters={"prompt": prompt},
            use_cache=True,  # cache workflow definition for 15 minutes
        )
