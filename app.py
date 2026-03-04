from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

print("Loading TinyLlama... (downloading ~2.2GB on first run)")

tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="./model")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir="./model",
    torch_dtype=torch.float32,
)

device = torch.device("cpu")
model.to(device)
model.eval()

print("TinyLlama loaded successfully!")

# Run a few sample prompts (no interactive input needed for CI)
prompts = [
    "What is artificial intelligence?",
    "Explain machine learning in simple terms.",
    "Write a short poem about the stars.",
]

messages_base = [
    {"role": "system", "content": "You are a helpful, friendly AI assistant."}
]

for user_input in prompts:
    print(f"\n{'='*50}")
    print(f"You: {user_input}")

    messages = messages_base + [{"role": "user", "content": user_input}]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    input_length = encoded["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **encoded,
            max_new_tokens=150,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = outputs[0][input_length:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

    print(f"AI: {response}")

print(f"\n{'='*50}")
print("TinyLlama inference complete!")
