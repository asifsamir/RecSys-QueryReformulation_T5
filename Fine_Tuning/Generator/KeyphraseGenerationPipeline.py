# Model parameters
import json

from transformers import (
    Text2TextGenerationPipeline,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
)


class KeyphraseGenerationPipeline(Text2TextGenerationPipeline):
    def __init__(self, model, model_name, keyphrase_sep_token=";", *args, **kwargs):
        super().__init__(
            model=AutoModelForSeq2SeqLM.from_pretrained(model),
            tokenizer=AutoTokenizer.from_pretrained(model_name),
            *args,
            **kwargs
        )
        self.keyphrase_sep_token = keyphrase_sep_token

    def postprocess(self, model_outputs):
        results = super().postprocess(
            model_outputs=model_outputs
        )
        return [[keyphrase.strip() for keyphrase in result.get("generated_text").split(self.keyphrase_sep_token) if keyphrase != ""] for result in results]


if __name__ == '__main__':
    # Load pipeline
    model_name = "ml6team/keyphrase-generation-t5-small-inspec"
    model_path = '../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230805_1319-4ep'
    generator = KeyphraseGenerationPipeline(model=model_path, model_name=model_name)


    # Replace 'path_to_trained_model' with the actual path to your trained model directory
    # model_loader = KeyphraseGenerator_T5(
    #     model_path='../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230805_1319-4ep')

    # read text file from path
    with open('../../Data/Cleaned_Data.json', 'r') as file:
        data = file.read()
        data = json.loads(data)

    # #read json into dictionary
    # import json
    # with open('../Data/test_data.json') as json_file:
    #     data = json.load(json_file)

    # convert the dictionary to dataframe
    import pandas as pd

    df = pd.DataFrame.from_dict(data)

    # print(df['bug_title'].head())
    # # Example input description
    # bug_description = "This is a bug description. Please see if AsicMiner.java is causing problem. shows 'NO_file_index'. What to do?."

    # Generate keyphrases using the loaded model
    print('Description:')
    bug_description = df['bug_description'][1]
    print(bug_description)

    generated_keyphrases = generator(bug_description,
                                     top_p=0.90,
                                     num_return_sequences=1,
                                     max_length=50,
                                     num_beams=6,
                                     # top_k=20
                                     )
    # Print the generated keyphrases
    print('Generated keyphrases:')
    print(generated_keyphrases)