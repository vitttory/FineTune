import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import importlib.metadata

_orig_version = importlib.metadata.version
def _patched_version(package_name):
    if package_name == "bitsandbytes":
        return "0.43.3" 
    return _orig_version(package_name)
importlib.metadata.version = _patched_version

MODEL_NAME = "microsoft/Phi-3-mini-4k-instruct"
NEW_MODEL_NAME = "phi-3-horse-riding-lora"
MERGED_DIR = "merged_model"

print("--- Starting Merge Process ---")

print("Reloading base model in FP16...")

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="cpu",
    trust_remote_code=True,
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

print(f"Loading adapter: {NEW_MODEL_NAME}...")
model = PeftModel.from_pretrained(base_model, NEW_MODEL_NAME)

print("Merging adapter into base model (this uses RAM)...")
model = model.merge_and_unload()

print(f"Saving merged model to '{MERGED_DIR}'...")

model.save_pretrained(MERGED_DIR, safe_serialization=True, max_shard_size="1GB")
tokenizer.save_pretrained(MERGED_DIR)

print("SUCCESS: Model merged and saved!")
print(f"Now you can convert the folder '{MERGED_DIR}' to GGUF format.")
