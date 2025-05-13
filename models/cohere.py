import base64
import os

import cohere

from .model import Model

COHERE_TEMPERATURE = 0.1


class CohereModel(Model):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = cohere.ClientV2(os.environ.get("COHERE_API_KEY"))

    def run(
        self,
        image: str,
        prompt: str,
        structured_output_format=None,
        image_name: str = "image.png",
    ):
        image_dtype = "image/" + image_name.split(".")[-1].replace("jpg", "jpeg")
        response = self.client.chat(
            model=self.model_id,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{image_dtype};base64,{base64.b64encode(image).decode()}"
                            },
                        },
                    ],
                }
            ],
            temperature=COHERE_TEMPERATURE,
        )

        return response.message.content[0].text
