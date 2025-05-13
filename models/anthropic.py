import base64

import anthropic

from .model import Model

ANTHROPIC_TEMPERATURE = 0.1


class AnthropicModel(Model):
    def __init__(self, model_id: str):
        self.model_id = model_id
        self.client = anthropic.Anthropic()

    def run(
        self,
        image: str,
        prompt: str,
        image_name="image.png",
        structured_output_format=None,
    ):
        message = self.client.messages.create(
            model=self.model_id,
            max_tokens=1024,
            temperature=ANTHROPIC_TEMPERATURE,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/"
                                + image_name.split(".")[-1].replace("jpg", "jpeg"),
                                "data": base64.b64encode(image).decode(),
                            },
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ],
        )

        return message.content[0].text
