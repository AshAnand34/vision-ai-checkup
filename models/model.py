class Model:
    def run(self, image: str, prompt: str, image_name=None):
        raise NotImplementedError("This method should be overridden by subclasses.")

    def run_with_retry(
        self,
        image: str,
        prompt: str,
        image_name=None,
        structured_output_format: str = None,
        retries: int = 3,
    ):
        for attempt in range(retries):
            try:
                return self.run(
                    image,
                    prompt,
                    image_name=image_name,
                    structured_output_format=structured_output_format,
                )
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying... model, {self.__class__.__name__}")
                    continue
                else:
                    print(f"All attempts failed: {e}")
                    return None
        return None
