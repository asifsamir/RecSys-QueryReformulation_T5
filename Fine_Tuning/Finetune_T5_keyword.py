import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import nltk
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load

class KeyphraseGenerationTrainer:
    def __init__(self, model_checkpoint, max_input_length=1024, max_target_length=60):
        self.model_checkpoint = model_checkpoint
        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
        self.keyphrase_sep_token = ';'
        self.metric = load("rouge")
        self.tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

        if model_checkpoint in ["t5-small", "t5-base", "t5-large", "t5-3b", "t5-11b", "Salesforce/codet5-small"]:
            self.prefix = "summarize: "
        else:
            self.prefix = ""
    @staticmethod
    def load_data(file_path):
        data = ''
        try:
            with open(file_path, 'r') as f:
                file_contents = f.read()
            data = json.loads(file_contents)
        except json.JSONDecodeError as e:
            print("Error decoding JSON:", e)
        except FileNotFoundError:
            print(f"File not found: '{file_path}'")
        except Exception as e:
            print("Error:", e)
        return data

    def preprocess_function(self, bug_description, reformed_query):
        document_inputs = self.tokenizer(
            bug_description,
            padding="max_length",
            truncation=True,
            max_length=self.max_input_length,
        )

        keyphrases = reformed_query.split()

        target_text = f" {self.keyphrase_sep_token} ".join(keyphrases)
        targets = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.max_target_length,
        )
        labels = targets.input_ids
        labels = [label if label != self.tokenizer.pad_token_id else -100 for label in labels]

        model_inputs = {
            "input_ids": document_inputs["input_ids"],
            "attention_mask": document_inputs["attention_mask"],
            "labels": labels,
        }

        return model_inputs

    def tokenize_data(self, dataset_df):
        return [self.preprocess_function(row["bug_description"], row["reformed_query"]) for _, row in dataset_df.iterrows()]

    def train(self, train_df, valid_df, batch_size=16, num_train_epochs=1, save=True):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)

        training_args = Seq2SeqTrainingArguments(
            f"{self.model_checkpoint.split('/')[-1]}-T5_keyphrase",
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=num_train_epochs,
            predict_with_generate=True,
            fp16=True,
        )

        data_collator = DataCollatorForSeq2Seq(self.tokenizer, model=model, label_pad_token_id=-100)

        tokenized_train_data = self.tokenize_data(train_df)
        tokenized_valid_data = self.tokenize_data(valid_df)

        trainer = Seq2SeqTrainer(
            model,
            args=training_args,
            train_dataset=tokenized_train_data,
            eval_dataset=tokenized_valid_data,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics
        )

        train_results = trainer.train()


        # Save the trained model and configuration
        if save:
            output_directory = f"../FineTunedModels/{self.model_checkpoint.split('/')[-1]}-T5_keyphrase-3ep"
            trainer.save_model(output_directory)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = self.metric.compute(predictions=decoded_preds, references=decoded_labels)
        result = {key: value * 100 for key, value in result.items()}
        prediction_lens = [np.count_nonzero(pred != self.tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        return {k: round(v, 4) for k, v in result.items()}

    def evaluate(self, trainer, tokenized_test_data, predict_with_generate=True, max_length=50, num_beams=3, early_stopping=True):
        if(type(tokenized_test_data) == pd.DataFrame):
            tokenized_test_data = self.tokenize_data(tokenized_test_data)

        predict_results = trainer.predict(tokenized_test_data, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
        metrics = predict_results.metrics
        print(metrics)

        if predict_with_generate:
            predictions = self.tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True,
                                                      clean_up_tokenization_spaces=True)
            predictions = [pred.strip() for pred in predictions]
            return predictions

    def generate(self, bug_description, reformed_query, model):
        model_inputs = self.preprocess_function(bug_description, reformed_query)
        input_ids = torch.tensor([model_inputs['input_ids']])
        attention_mask = torch.tensor([model_inputs['attention_mask']])
        outputs = model.generate(input_ids, attention_mask=attention_mask)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Load data
    file_path = '../Data/Cleaned_newLine_Data.json'
    data = KeyphraseGenerationTrainer.load_data(file_path)
    dataset_df = pd.DataFrame.from_dict(data)

    # Split data into train, test, validation
    train_df, test_valid_df = train_test_split(dataset_df, test_size=0.15, random_state=42)
    valid_df, test_df = train_test_split(test_valid_df, test_size=0.35, random_state=42)

    # save test_df to json
    test_df.to_json('../Data/test_data.json', orient='records', lines=True)


    # Train the model
    trainer = KeyphraseGenerationTrainer(model_checkpoint="ml6team/keyphrase-generation-t5-small-inspec", max_input_length=1024, max_target_length=60)
    trainer.train(train_df, valid_df, batch_size=16, num_train_epochs=1, save=True)

    # Evaluate with test data and get predictions
    predictions = trainer.evaluate(tokenized_test_data=test_df, predict_with_generate=True, max_length=50, num_beams=3, early_stopping=True)
    print(predictions[:5])  # Print the first 5 predictions
