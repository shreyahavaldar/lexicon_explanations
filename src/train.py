from transformers import TrainingArguments, Trainer
import numpy as np
import evaluate
import os
import torch
from sklearn.metrics import f1_score, mean_squared_error, accuracy_score, balanced_accuracy_score, classification_report
from torch.nn import BCEWithLogitsLoss
import glob


def train(config, pipeline, train_data, val_data, data_test, batch_size=8, lr=1e-5):
    dataset_name = config["dataset"]
    model_name = pipeline.model.__class__.__name__
    print(model_name)
    log_dir = f"checkpoints/{dataset_name}_{model_name}"
    resume = False
    if glob.glob(log_dir + "/checkpoint-*"): #os.path.exists(log_dir + "/checkpoint-500"):
        resume = True

    if os.path.exists(log_dir + "/pytorch_model.bin"):
        pipeline.model = pipeline.model.from_pretrained(log_dir).cuda()
        # return

    training_args = TrainingArguments(
        output_dir=log_dir,
        num_train_epochs=3,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        learning_rate=lr,
        evaluation_strategy="steps",
        fp16=True,
        save_total_limit=1)
    if dataset_name == "emobank" or dataset_name == "polite":
        metric = evaluate.load("mse")
    else:
        metric = evaluate.load("accuracy")

    def tokenize_function(examples):
        return pipeline.tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=512)

    train_data_tokenized = train_data.map(tokenize_function, batched=True)
    val_data_tokenized = val_data.map(tokenize_function, batched=True)
    # print(train_data_tokenized[0])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        if dataset_name == "goemotions" or dataset_name == "blog":
            pred = torch.from_numpy(logits).float().sigmoid() > 0.5
            print(classification_report(labels.astype(int), pred.int()))
            return {"f1-average": f1_score(labels, pred, average='weighted'),
                    "micro-accuracy": accuracy_score(labels, pred)}
        elif dataset_name != "emobank" and dataset_name != "polite":
            predictions = np.argmax(logits, axis=-1)
        else:
            predictions = logits

        # For the yelp dataset, we calculate full metrics and polarity metrics
        if dataset_name == "yelp":
            pred_binary = predictions > 1
            labels_binary = labels > 1
            return {"f1": f1_score(labels, predictions, average='weighted'),
                    "accuracy": accuracy_score(labels, predictions),
                    "polarity-f1": f1_score(labels_binary, pred_binary),
                    "polarity-accuracy": accuracy_score(labels_binary, pred_binary)}
        else:
            return metric.compute(predictions=predictions, references=labels)

    # if dataset_name == "goemotions" or dataset_name == "blog":
    #     all_labels = torch.tensor([row["labels"] for row in train_data_tokenized])
    #     weights = (all_labels == 0.).sum() / torch.sum(all_labels, dim=0)
    #     print("Weights:", weights)
    #     class CustomTrainer(Trainer):
    #         def compute_loss(self, model, inputs, return_outputs=False):
    #             labels = inputs.get("labels")
    #             # forward pass
    #             outputs = model(**inputs)
    #             logits = outputs.get('logits')
    #             # compute custom loss
    #             loss_fct = BCEWithLogitsLoss(pos_weight=weights.cuda())
    #             loss = loss_fct(logits.view(-1, self.model.config.num_labels),
    #                             labels.view(-1, self.model.config.num_labels))
    #             return (loss, outputs) if return_outputs else loss

    #     trainer = CustomTrainer(
    #         model=pipeline.model,
    #         args=training_args,
    #         train_dataset=train_data_tokenized,
    #         eval_dataset=val_data_tokenized,
    #         compute_metrics=compute_metrics,
    #     )
    # else:
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
        print(trainer.predict(data_test))
    else:
        trainer.train(resume_from_checkpoint=resume)
        trainer.save_model()