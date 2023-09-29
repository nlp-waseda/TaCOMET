import argparse
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer
)

NEED_TOKEN = 'xNeed'
EFFECT_TOKEN = 'xEffect'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--graph_jsonl', default='data/mrph_train.jsonl')
    parser.add_argument('--model_name_or_path', default='nlp-waseda/gpt2-xl-japanese')
    parser.add_argument('--from_comet', action="store_true", default=False)
    parser.add_argument('--output_dir', default="./")
    parser.add_argument('--max_length', default=128, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--auto_find_bs', action="store_true", default=False)
    parser.add_argument('--learning_rate', default=2e-5, type=float)
    parser.add_argument('--num_epochs', default=3, type=int)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--deepspeed", type=str, default="./ds_config.json")
    args = parser.parse_args()
    

    raw_datasets = load_dataset('json', data_files=args.graph_jsonl, split='train')

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    
    if not args.from_comet:
        special_tokens_dict = {
            'additional_special_tokens': [
                NEED_TOKEN,
                EFFECT_TOKEN,
            ]
        }
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    else:
        num_added_toks = 4


    def preprocess_function(examples):
        outputs = []
        
        for head, rel_infs in zip(examples['head'], examples['inferences']):
            for rel, time_inf in rel_infs.items():
                if time_inf == None:
                    continue
                
                for time, inf in time_inf.items():
                    if inf == None:
                        continue
                    
                    if rel == "xNeed":
                        conj = " 前 、 "
                    else:
                        conj = " 後 、 "
                    outputs.append(head+" "+rel+" "+time+conj+inf+tokenizer.eos_token)

        return {'data': outputs}


    preprocessed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_datasets.column_names,
    )

    tokenized_dataset_unsplitted = preprocessed_datasets.map(
        lambda examples: tokenizer(
            examples['data'],
            truncation=True,
            max_length=args.max_length,
        ),
        batched=True,
        remove_columns=preprocessed_datasets.column_names,
    )
    
    tokenized_datasets = tokenized_dataset_unsplitted.train_test_split(test_size=0.05, shuffle=True, seed=42)

    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        evaluation_strategy='epoch',
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        num_train_epochs=args.num_epochs,
        save_total_limit=args.num_epochs,
        logging_steps=100 ,
        logging_strategy='steps',
        save_strategy='epoch',
        deepspeed=args.deepspeed,
        auto_find_batch_size=args.auto_find_bs,
    )

    class DataCollatorForComet(DataCollatorForLanguageModeling):
        def torch_call(self, examples):
            batch = super().torch_call(examples)            
            labels = batch['labels']

            # consider eos
            eos_mask = labels == -100
            eos_mask[:, 1:] = eos_mask[:, 1:] ^ eos_mask[:, :-1]
            labels[eos_mask] = self.tokenizer.eos_token_id

            # ignore h and r
            rel_mask = labels >= len(tokenizer)-num_added_toks
            tail_mask = (rel_mask.cumsum(dim=-1) - rel_mask.to(int)).to(bool)
            
            # Find the first occurrence of the comma token ('、') after the relation token
            comma_mask = (labels == self.tokenizer.convert_tokens_to_ids('、')) & tail_mask
            tail_mask = (comma_mask.cumsum(dim=-1) - comma_mask.to(int)).to(bool)

            labels[~tail_mask] = -100

            batch['labels'] = labels
            return batch

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForComet(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['test'],
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    trainer.save_model()
    trainer.save_state()


if __name__ == '__main__':
    main()
