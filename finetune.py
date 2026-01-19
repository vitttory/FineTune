import torch
import matplotlib.pyplot as plt
from datasets import load_dataset

import importlib.metadata
_orig_version = importlib.metadata.version
def _patched_version(package_name):
    if package_name == "bitsandbytes":
        return "0.43.3" 
    return _orig_version(package_name)
importlib.metadata.version = _patched_version
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
NEW_MODEL_NAME = "phi-3-horse-riding-lora"

DATA_FILE = "train.jsonl"
OUTPUT_DIR = "./results"
PLOT_LOSS_FILE = "plot_loss.png"
PLOT_ACC_FILE = "plot_accuracy.png"

NUM_EPOCHS = 10
BATCH_SIZE = 1
GRADIENT_ACCUMULATION = 4
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512

def format_instruction(sample):
    """
    Formats the JSONL entry into the Phi-3 chat format.
    """
    return f"<|user|>\n{sample['instruction']}<|end|>\n<|assistant|>\n{sample['output']}<|end|>"

print(f"Loading dataset from {DATA_FILE}...")
dataset = load_dataset("json", data_files=DATA_FILE, split="train")

# Split dataset: 90% train, 10% test
print("Splitting dataset into Train (90%) and Validation (10%)...")
dataset = dataset.train_test_split(test_size=0.1)
print(f"Train size: {len(dataset['train'])}")
print(f"Val size:   {len(dataset['test'])}")

print("Sample entry format:")
print(format_instruction(dataset['train'][0]))

def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = preds[:, :-1]
    labels = labels[:, 1:]
    mask = labels != -100

    filtered_preds = preds[mask]
    filtered_labels = labels[mask]
    
    if len(filtered_labels) == 0:
        return {"accuracy": 0.0}
        
    accuracy = (filtered_preds == filtered_labels).mean()
    
    return {"accuracy": accuracy}

print(f"Loading base model: {MODEL_NAME}...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=False,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.1,
    r=16, 
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_length=MAX_SEQ_LENGTH,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION,
    optim="paged_adamw_32bit",

    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    
    learning_rate=LEARNING_RATE,
    weight_decay=0.001,
    fp16=False,
    bf16=False,
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="constant",
    report_to="none",

    dataset_text_field="text",
    packing=False
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
    peft_config=peft_config,
    formatting_func=format_instruction,
    processing_class=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
)

print("Starting training...")
trainer_stats = trainer.train()

print("Training finished!")
print(f"Final Loss: {trainer_stats.training_loss}")

trainer.model.save_pretrained(NEW_MODEL_NAME)
print(f"Model saved to {NEW_MODEL_NAME}")

print("Generating training graphs...")

history = trainer.state.log_history

epochs = []
train_losses = []
eval_losses = []
eval_accuracies = []

epoch_data = {}

for entry in history:
    if 'epoch' in entry:
        e = entry['epoch']
        if e not in epoch_data:
            epoch_data[e] = {}
        
        if 'loss' in entry:
            epoch_data[e]['train_loss'] = entry['loss']
        
        if 'eval_loss' in entry:
            epoch_data[e]['eval_loss'] = entry['eval_loss']
            
        if 'eval_accuracy' in entry:
            epoch_data[e]['eval_accuracy'] = entry['eval_accuracy']

sorted_epochs = sorted(epoch_data.keys())
final_epochs = []
final_train_loss = []
final_eval_loss = []
final_eval_acc = []

for e in sorted_epochs:
    data = epoch_data[e]

    if 'train_loss' in data or 'eval_loss' in data:
        final_epochs.append(e)
        final_train_loss.append(data.get('train_loss', None))
        final_eval_loss.append(data.get('eval_loss', None))
        final_eval_acc.append(data.get('eval_accuracy', None))

plt.figure(figsize=(10, 6))

t_epochs = [e for e, l in zip(final_epochs, final_train_loss) if l is not None]
t_loss = [l for l in final_train_loss if l is not None]
v_epochs = [e for e, l in zip(final_epochs, final_eval_loss) if l is not None]
v_loss = [l for l in final_eval_loss if l is not None]

if t_epochs:
    plt.plot(t_epochs, t_loss, label='Training Loss', color='blue', marker='o')
if v_epochs:
    plt.plot(v_epochs, v_loss, label='Validation Loss', color='orange', marker='x')

plt.title(f'Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(PLOT_LOSS_FILE)
print(f"Loss plot saved to {PLOT_LOSS_FILE}")

plt.figure(figsize=(10, 6))
a_epochs = [e for e, a in zip(final_epochs, final_eval_acc) if a is not None]
a_acc = [a for a in final_eval_acc if a is not None]

if a_epochs:
    plt.plot(a_epochs, a_acc, label='Validation Accuracy', color='green', marker='s')
    plt.title(f'Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig(PLOT_ACC_FILE)
    print(f"Accuracy plot saved to {PLOT_ACC_FILE}")
else:
    print("No accuracy data found to plot.")

print("Done! Now run 'python merge_adapter.py' to create the merged model for GGUF conversion.")
