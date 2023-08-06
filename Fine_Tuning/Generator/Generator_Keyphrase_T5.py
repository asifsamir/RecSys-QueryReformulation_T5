import json

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


class KeyphraseGenerator_T5:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained('ml6team/keyphrase-generation-t5-small-inspec')
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def generate_keyphrases(self, bug_description):
        # inputs = self.tokenizer(bug_description, return_tensors="pt")
        # outputs = self.model.generate(inputs.input_ids, max_length=50, num_beams=15, topP=0.90, early_stopping=True)
        # keyphrases = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        # Create a text generation pipeline
        generator = pipeline('text-generation', model=self.model, tokenizer=self.tokenizer)

        # Generate text
        keyphrases = generator(bug_description,
                               max_length=50,
                               top_p=0.90,
                               num_return_sequences=1)

        return keyphrases[0]['generated_text']

# Example usage:
if __name__ == "__main__":
    # Replace 'path_to_trained_model' with the actual path to your trained model directory
    model_loader = KeyphraseGenerator_T5(model_path='../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230805_1319-4ep')

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


    generated_keyphrases = model_loader.generate_keyphrases(bug_description)
    # Print the generated keyphrases
    print('Generated keyphrases:')
    print(generated_keyphrases)
