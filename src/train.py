from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import os
import torch
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, balanced_accuracy_score
from torch.nn import BCEWithLogitsLoss


def train(config, pipeline, train_data, val_data, batch_size=8, lr=1e-5):
    dataset_name = config["dataset"]
    model_name = pipeline.model.__class__.__name__
    print(model_name)
    log_dir = f"checkpoints/{dataset_name}_{model_name}"
    resume = False
    if os.path.exists(log_dir + "/checkpoint-500"):
        resume = True

    if os.path.exists(log_dir + "/pytorch_model.bin"):
        pipeline.model = pipeline.model.from_pretrained(log_dir).cuda()
        # return

    training_args = TrainingArguments(
        output_dir=log_dir,
        evaluation_strategy="epoch",
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=16,
        learning_rate=lr,
        weight_decay=5e-8,
        save_total_limit=1)
    if dataset_name == "emobank" or dataset_name == "polite":
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    def tokenize_function(examples):
        return pipeline.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=512)

    train_data_tokenized = train_data.map(tokenize_function, batched=True)
    val_data_tokenized = val_data.map(tokenize_function, batched=True)
    print(train_data_tokenized[0])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if dataset_name == "goemotions" or dataset_name == "blog":
            pred = torch.from_numpy(logits).sigmoid() > 0.5
            return {"f1-average": f1_score(labels, pred, average='weighted'), "micro-accuracy": accuracy_score(labels, pred), "macro-accuracy": balanced_accuracy_score(labels, pred)}
        elif dataset_name != "emobank" and dataset_name != "polite":
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions = logits

        return metric.compute(predictions=predictions, references=labels)

    if dataset_name == "goemotions" or dataset_name == "blog":
        y = torch.stack([torch.from_numpy(row["labels"]) for row in train_data_tokenized])
        print("All label shape:", y.shape)
        weights = (y == 0.) / torch.sum(y, dim=0)
        class CustomTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False):
                labels = inputs.get("labels")
                # forward pass
                outputs = model(**inputs)
                logits = outputs.get('logits')
                # compute custom loss
                loss_fct = BCEWithLogitsLoss(weight=weights)
                loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
                return (loss, outputs) if return_outputs else loss

        trainer = CustomTrainer(
            model=pipeline.model,
            args=training_args,
            train_dataset=train_data_tokenized,
            eval_dataset=val_data_tokenized,
            compute_metrics=compute_metrics,
        )
    else:
        trainer = Trainer(
            model=pipeline.model,
            args=training_args,
            train_dataset=train_data_tokenized,
            eval_dataset=val_data_tokenized,
            compute_metrics=compute_metrics,
        )

    # # Train only the classification layer
    # for name, param in pipeline.model.named_parameters():
    #     print(name)
    #     if 'classifier' not in name: # classifier layer
    #         param.requires_grad = False

    if os.path.exists(log_dir + "/pytorch_model.bin"):
        print(trainer.evaluate())
    else:
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model()