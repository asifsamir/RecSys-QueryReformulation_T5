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
            tokenizer=AutoTokenizer.from_pretrained(model_name, max_length=1024, truncation=True, padding=True),
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
    model_path = '../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230806_0305-50ep'
    generator = KeyphraseGenerationPipeline(model=model_path, model_name=model_name)


    # read text file from path
    with open('../../Data/Augmented/Train_Test/test.json', 'r') as file:
        data = file.read()
        data = json.loads(data)

    import pandas as pd

    df = pd.DataFrame.from_dict(data)

    print('Suggested keyphrases:')
    print(df['reformed_query'][1])

    print('\nDescription:')
    bug_description = df['bug_description'][1]
    print(bug_description)


    dict_config_output = []


    generated_keyphrases = generator(bug_description,
                                     # top_p=0.5,
                                     num_return_sequences=5,
                                     max_length=50,
                                     num_beams=15,
                                     # no_repeat_ngram_size=2,
                                     # temperature=0.3,
                                     do_sample=True,
                                     # top_k=40
                                     )
    # Print the generated keyphrases
    print('Generated keyphrases:')
    for key_phrase in generated_keyphrases:
        print(key_phrase)