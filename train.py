import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def train(data_file, output_dir):
    # 配置、分词器和模型加载
    config = GPT2Config.from_pretrained("gpt2", output_hidden_states=False)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=config)

    # 数据集准备
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=data_file,
        block_size=128
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # 训练参数设置
    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2
    )

    # 初始化训练器并进行训练
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, required=True, help='Path to the training data file.')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save the fine-tuned model.')
    args = parser.parse_args()
    train(args.data_file, args.output_dir)
