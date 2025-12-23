
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = r"model"

def load_tokenizer_and_model(model_path):
    print(f"Загружаю модель из локальной папки: {model_path}")
    return GPT2Tokenizer.from_pretrained(model_path), GPT2LMHeadModel.from_pretrained(model_path)

def generate(model, tok, text, **kwargs):
    input_ids = tok.encode(text, return_tensors="pt")
    out = model.generate(input_ids, **kwargs)
    return [tok.decode(o, skip_special_tokens=True) for o in out]

# Загрузка модели
tok, model = load_tokenizer_and_model(MODEL_PATH)

# промпт
prompt = "Ты не воспринимаешь мои слова серьезно, это не смешно, я тебе скажу кое-что серьезное,еще раз повторяю,я не"

print(f"Промпт: {prompt}")

# Генерация
generated = generate(
    model, tok, prompt,
    do_sample=True,
    max_length=150,
    repetition_penalty=1.2,
    top_k=50,
    top_p=0.95,
    temperature=0.5,
    #num_beams=None,
    no_repeat_ngram_size=4
)

print("\n" + "="*50)
print("Сгенерированный текст:")
print(generated[0])



