from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import os
import torch
from sklearn.metrics import f1_score, mean_squared_error

def train(config, pipeline, train_data, val_data):
    dataset_name = config["dataset"]
    model_name = pipeline.model.__class__.__name__
    print(model_name)
    log_dir = f"checkpoints/{dataset_name}_{model_name}"
    resume = False
    if os.path.exists(log_dir + "/checkpoint-500"):
        resume = True

    if os.path.exists(log_dir + "/pytorch_model.bin"):
        pipeline.model = pipeline.model.from_pretrained(log_dir).cuda()
        return

    training_args = TrainingArguments(output_dir=log_dir, evaluation_strategy="epoch")
    if dataset_name == "emobank" or dataset_name == "polite":
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    def tokenize_function(examples):
        return pipeline.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=256)

    train_data_tokenized = train_data.map(tokenize_function, batched=True)
    val_data_tokenized = val_data.map(tokenize_function, batched=True)
    print(train_data_tokenized[0])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if dataset_name == "goemotions" or dataset_name == "blog":
            pred = torch.from_numpy(logits).sigmoid() > 0.5
            return {"f1-average": f1_score(pred, labels, average='weighted')}
        elif dataset_name != "emobank" and dataset_name != "polite":
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions = logits

        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=pipeline.model,
        args=training_args,
        train_dataset=train_data_tokenized,
        eval_dataset=val_data_tokenized,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model()