import telebot
import torch
import importlib.metadata
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

_orig_version = importlib.metadata.version
def _patched_version(package_name):
    if package_name == "bitsandbytes":
        return "0.43.3" 
    return _orig_version(package_name)
importlib.metadata.version = _patched_version

TELEGRAM_BOT_TOKEN = '7948674171:AAHx4EaIe7LdVrjiH_k8SlnzauXC_P0aSUE'
MODEL_PATH = "./merged_model"
SYSTEM_PROMPT = "You are a riding instructor teaching a rider IN THE SADDLE. Focus ONLY on aids: Seat, Legs, Reins. Do not explain theory. Be extremely short."

print(f"Loading model from {MODEL_PATH}...")

try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        quantization_config=bnb_config,
        device_map="auto"
    )
    
    print("Model loaded successfully!")

except Exception as e:
    print(f"CRITICAL ERROR: {e}")
    exit(1)

def generate_response(user_input):
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input},
        ]

        input_ids = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(model.device)

        outputs = model.generate(
            input_ids,
            max_new_tokens=60,
            temperature=0.1,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

    except Exception as e:
        print(f"Error: {e}")
        return "Error."

bot = telebot.TeleBot(TELEGRAM_BOT_TOKEN)

@bot.message_handler(commands=['start'])
def send_welcome(message):
    bot.reply_to(message, "Hello! I am your AI Horse Training Assistant. Ask me anything about riding! 🐴")


@bot.message_handler(func=lambda message: True)
def handle_message(message):
    print(f"User: {message.text}")
    bot.send_chat_action(message.chat.id, 'typing')

    response = generate_response(message.text)

    bot.reply_to(message, response)
    print(f"Bot: {response}")


if __name__ == "__main__":
    print("Bot is starting...")
    bot.infinity_polling()