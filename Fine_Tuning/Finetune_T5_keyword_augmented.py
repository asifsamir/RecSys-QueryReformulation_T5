from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
import numpy as np
from transformers import AutoTokenizer
import torch
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
from evaluate import load
import tensorflow as tf

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

    def train(self, train_df, valid_df, batch_size=16, epochs_train=1, save=True):
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_checkpoint)

        training_args = Seq2SeqTrainingArguments(
            f"{self.model_checkpoint.split('/')[-1]}-T5_keyphrase",
            learning_rate=2e-5,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=epochs_train,
            predict_with_generate=True,
            # fp16=True,
            report_to="tensorboard"
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
            current_date_time = datetime.now().strftime("%Y%m%d_%H%M")
            # output_directory = f"../Models_Fine_Tuned/{self.model_checkpoint.split('/')[-1]}-T5_keyphrase-3ep"
            # use date and time and epochs to name the output directory
            output_directory = f"../Models_Fine_Tuned/{self.model_checkpoint.split('/')[-1]}-{current_date_time}-{epochs_train}ep"
            trainer.save_model(output_directory)
            
        
        return trainer, train_results


        

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
    


    # def evaluate(self, trainer, tokenized_test_data, predict_with_generate=True, max_length=50, num_beams=3, early_stopping=True):
    #     # if(type(tokenized_test_data) == pd.DataFrame):
    #     if isinstance(tokenized_test_data, pd.DataFrame):
    #         tokenized_test_data = self.tokenize_data(tokenized_test_data)

    #     predict_results = trainer.predict(tokenized_test_data, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
    #     metrics = predict_results.metrics
    #     print(metrics)

    #     if predict_with_generate:
    #         predictions = self.tokenizer.batch_decode(predict_results.predictions, skip_special_tokens=True,
    #                                                   clean_up_tokenization_spaces=True)
    #         predictions = [pred.strip() for pred in predictions]
    #         return predictions

    def evaluate(self, trainer, tokenized_test_data, max_length=50, num_beams=3, early_stopping=True):
        if isinstance(tokenized_test_data, pd.DataFrame):
            tokenized_test_data = self.tokenize_data(tokenized_test_data)

        predictions = trainer.predict(
            test_dataset=tokenized_test_data,  # Rename tokenized_test_data to test_dataset
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping
        )

        # predictions = trainer.predict(tokenized_test_data, max_length=max_length, num_beams=num_beams, early_stopping=early_stopping)
        
        metrics = predictions.metrics
        print(metrics)

        if trainer.args.predict_with_generate:  # Use trainer.args.predict_with_generate
            decoded_preds = self.tokenizer.batch_decode(
                predictions.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            decoded_preds = [pred.strip() for pred in decoded_preds]
            return decoded_preds
    

    def generate(self, bug_description, reformed_query, model):
        model_inputs = self.preprocess_function(bug_description, reformed_query)
        input_ids = torch.tensor([model_inputs['input_ids']])
        attention_mask = torch.tensor([model_inputs['attention_mask']])
        outputs = model.generate(input_ids, attention_mask=attention_mask)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    @classmethod
    def prepare_data(cls, dataset_df):
        # create an empty dataframe to return
        prepared_df = pd.DataFrame(columns=['reformed_query', 'bug_description', 'bug_title', 'bug_id', 'repo'])

        # iterate over the dataframe
        for index, row in dataset_df.iterrows():
            # get the list from 'effective_queries' column and iterate over it
            for i in range(len(row['effective_queries'])):
                # add a row to the prepared_df dataframe.
                prepared_df.loc[len(prepared_df)] = [row['effective_queries'][i], row['bug_description'], row['bug_title'], row['bug_id'], row['repo']]
        return prepared_df



if __name__ == "__main__":
    import os
    os.environ["WANDB_DISABLED"] = "true"
               
    # Load data
    file_path = '../Data/Augmented/cleaned_effective.json'
    data = KeyphraseGenerationTrainer.load_data(file_path)
    dataset_df = pd.DataFrame.from_dict(data)

    # Prepare data: creating a dataframe with each row containing a query, bug_description, bug_title, bug_id, repo from the original data
    dataset_df = KeyphraseGenerationTrainer.prepare_data(dataset_df)

    print(dataset_df.head())
    print(dataset_df.shape)

    # Split data into train, test, validation
    train_df, test_valid_df = train_test_split(dataset_df, test_size=0.2, random_state=42, shuffle=True)
    valid_df, test_df = train_test_split(test_valid_df, test_size=0.6, random_state=42, shuffle=True)

    # save test_df to json
    test_df.to_json('../Data/test_data.json', orient='records', lines=True)


    # Train the model
    trainer_class = KeyphraseGenerationTrainer(model_checkpoint="ml6team/keyphrase-generation-t5-small-inspec", max_input_length=1024, max_target_length=60)
    trainer , train_results = trainer_class.train(train_df, valid_df, batch_size=8, epochs_train=10, save=True)

    # Evaluate with test data and get predictions
    # predictions = trainer.evaluate(trainer= tokenized_test_data=test_df, predict_with_generate=True, max_length=50, num_beams=3, early_stopping=True)
    # predictions = trainer.evaluate(trainer=trainer, tokenized_test_data=test_df, predict_with_generate=True, max_length=50, num_beams=3, early_stopping=True)
    predictions = trainer_class.evaluate(
        trainer=trainer,
        tokenized_test_data=test_df,
        max_length=50,
        num_beams=3,
        early_stopping=True
    )
    print(predictions[:5])  # Print the first 5 predictions
