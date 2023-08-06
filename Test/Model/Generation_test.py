from Fine_Tuning.Generator.KeyphraseGenerationPipeline import KeyphraseGenerationPipeline
import json

if __name__ == '__main__':
    # Load pipeline
    model_name = "ml6team/keyphrase-generation-t5-small-inspec"
    model_path = '../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230806_0305-50ep'
    generator = KeyphraseGenerationPipeline(model=model_path, model_name=model_name)


    # Replace 'path_to_trained_model' with the actual path to your trained model directory
    # model_loader = KeyphraseGenerator_T5(
    #     model_path='../../Models_Fine_Tuned/keyphrase-generation-t5-small-inspec-20230805_1319-4ep')

    # read text file from path
    with open('../../Data/Augmented/Train_Test/test.json', 'r') as file:
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
    print('Suggested keyphrases:')
    print(df['reformed_query'][0])

    print('\nDescription:')
    bug_description = df['bug_description'][0]
    print(bug_description)


    dict_config_output = []

    # # for loop which will update in interval of 3 upto 50
    # for i in range(3, 50, 3):
    #     for j in range(.5, .95, .05):
    #         generated_keyphrases = generator(bug_description,
    #                                          top_p=0.90,
    #                                          num_return_sequences=1,
    #                                          max_length=50,
    #                                          num_beams=i,
    #                                          no_repeat_ngram_size=2,
    #                                          # top_k=20
    #                                          )
    #         # Print the generated keyphrases
    #         print(f'Generated keyphrases: {i}')
    #         print(generated_keyphrases)
    #
    #         dict_config_output.append({'num_beams': i, 'top_p': j, 'keyphrases': generated_keyphrases})
    #
    #
    # # convert the dictionary to dataframe
    # pd = pd.DataFrame.from_dict(dict_config_output)


    generated_keyphrases = generator(bug_description,
                                     top_p=0.95,
                                     num_return_sequences=5,
                                     max_length=50,
                                     # num_beams=15,
                                     # no_repeat_ngram_size=2,
                                     # temperature=0.6,
                                     do_sample=True,
                                     top_k=50
                                     )
    # Print the generated keyphrases
    print('Generated keyphrases:')
    for key_phrase in generated_keyphrases:
        print(key_phrase)
    # print(generated_keyphrases)