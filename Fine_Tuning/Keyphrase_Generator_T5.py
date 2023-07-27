from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class KeyphraseGenerator_T5:
    def __init__(self, model_path):
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

    def generate_keyphrases(self, bug_description):
        inputs = self.tokenizer(bug_description, return_tensors="pt")
        outputs = self.model.generate(inputs.input_ids, max_length=50, num_beams=3, early_stopping=True)
        keyphrases = self.tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        return keyphrases

# Example usage:
if __name__ == "__main__":
    # Replace 'path_to_trained_model' with the actual path to your trained model directory
    model_loader = KeyphraseGenerator_T5(model_path='../FineTunedModels/T5_keyphrase-3ep')

    # Example input description
    bug_description = "This is a bug description. Please see if AsicMiner.java is causing problem. shows 'NO_file_index'. What to do?."

    # Generate keyphrases using the loaded model
    generated_keyphrases = model_loader.generate_keyphrases(bug_description)

    # Print the generated keyphrases
    print(generated_keyphrases)
