import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

SAMPLES = [
    "এই সিনেমাটি আমার খুব ভালো লেগেছে।",
    "বাংলাদেশ দক্ষিণ এশিয়ার একটি দেশ।",
    "আজকের সংবাদটি সত্য কিনা তা যাচাই করা প্রয়োজন।",
]


def verify_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    print(f"\nModel: {model_id}")
    print(f"Tokenizer: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {len(tokenizer)}")
    print(f"Chat template: {tokenizer.chat_template is not None}")

    for text in SAMPLES:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        decoded = tokenizer.decode(tokens)

        print(f"\nText: {text}")
        print(f"Tokens: {len(tokens)}")
        print(f"Characters: {len(text)}")
        print(f"Chars/token: {len(text) / len(tokens):.2f}")
        print(f"Decoded: {decoded}")

    return tokenizer


def verify_generation(model_id, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    messages = [
        {
            "role": "user",
            "content": (
                "নিচের বাক্যের অনুভূতি নির্ধারণ করুন। "
                "শুধু Positive, Negative, অথবা Neutral লিখুন।\n\n"
                "বাক্য: এই সিনেমাটি আমার খুব ভালো লেগেছে।"
            ),
        }
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

    generated = output[0, inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()

    print(f"\nGeneration test: {response}")

    del model
    torch.cuda.empty_cache()


def main():
    for model_id in MODELS:
        tokenizer = verify_tokenizer(model_id)

        if torch.cuda.is_available():
            verify_generation(model_id, tokenizer)
        else:
            print("\nGPU unavailable. Generation test skipped.")


if __name__ == "__main__":
    main()