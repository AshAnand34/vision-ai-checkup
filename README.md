# Vision AI Checkup

![](./images/screenshot.png)

[Vision AI Checkup](https://visioncheckup.com) is a tool to evaluate vision-capable language models on real-world problems.

The assessment consists of dozens of images, questions, and answers that we benchmark against models. We run the assessment every time we add a new model to the leaderboard.

You can use the vision assessment to gauge how well a model does generally, without having to understand a complex benchmark with thousands of data points.

The assessment and models are constantly evolving. This means that as more tasks get added or models receive updates, we can build a clearer picture of the current state-of-the-art models in real-time.

## Run the Assessment

To run the assessment suite, first clone this project and install the required dependencies:

```
git clone https://github.com/roboflow/vision-ai-checkup
pip install -r requirements.txt
```

You will then need API keys for all vendors used in the assessment. You can set these as follows:

```
export OPENAI_API_KEY=your-key
export HUGGINGFACE_API_KEY=your-key # used for Llama
export ANTHROPIC_API_KEY=your-key
export CO_API_KEY=your-key # you will need a production API key
```

Then, run:

```
python3 assess.py
```

## Contributing

### Contribute an Assessment

You can contribute an assessment to add to the Vision AI assessment suite.

To contribute an assessment, first clone this project:

```
git clone https://github.com/roboflow/vision-ai-checkup
```

Then:

1. Add the image you want to use in your assessment to the `images` folder.
2. Add an entry to the `prompts.csv` file with:
    - The file name (`file_name`).
    - The prompt to use (`prompt`).
    - The correct answer (`answer`).
    - A name for the assessment (`assessment_name`).
    - A category for the assessment. If possible, choose an existing category. If your assessment requires a new category, please note why this makes sense in your PR description.
    - Your name (`submitted_by`).
    - A URL you want to link to (`submitted_by_link`).
3. File a PR.

> [!WARNING]
> 
> Images must be no more than 2MB. This will ensure that your image is not too big to be run through supported APIs.

### Add a Model

The `models` directory lists all of the supported model vendors.

If you want to add a new model by a supported vendor, update the `model_providers` dictionary in the `assess.py` file and add the model ID.

If you want to add a model by a vendor that is not yet supported, create a file in the `models` directory following the same structure as the other models.

### Bugs, other changes

If you notice any bugs or see improvements that can be made to the assessment code or website, please create an Issue so we can discuss the changes before you start work.

## License

This project is licensed under an [MIT license](LICENSE).