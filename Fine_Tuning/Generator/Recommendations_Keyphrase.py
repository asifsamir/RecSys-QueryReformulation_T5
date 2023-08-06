import json

from Fine_Tuning.Generator.KeyphraseGenerationPipeline import KeyphraseGenerationPipeline


class Recommmender():
    model_name = "ml6team/keyphrase-generation-t5-small-inspec"
    model_path = '../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230805_1319-4ep'
    def __init__(self, model_name=None, model_path=None):

        if model_name is not None:
            self.model_name = model_name
        if model_path is not None:
            self.model_path = model_path

        self.generator = KeyphraseGenerationPipeline(model=self.model_path, model_name=self.model_name)


    def get_recommendations(self, bug_description, num_of_recommendations=1):
        recommendations = self.generator(bug_description,
                                         top_p=0.95,
                                         num_return_sequences=5,
                                         max_length=50,
                                         # num_beams=15,
                                         # no_repeat_ngram_size=2,
                                         # temperature=0.6,
                                         do_sample=True,
                                         top_k=50
                                         )
        return recommendations


if __name__ == '__main__':
    # Load pipeline
    model_name = "ml6team/keyphrase-generation-t5-small-inspec"
    model_path = '../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230805_1319-4ep'

    recommender = Recommmender()

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


    generated_keyphrases = recommender.get_recommendations(bug_description, num_of_recommendations=1)

    print('Generated keyphrases:', generated_keyphrases)