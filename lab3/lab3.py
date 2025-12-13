import torch
from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F

# Инициализация модели
model_path = r"bert-base-multilingual-uncased"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForMaskedLM.from_pretrained(model_path, return_dict=True)
model.eval()


def predict_masked_words(sentence_with_mask, top_k=10):

    inputs = tokenizer.encode_plus(
        sentence_with_mask,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    mask_token_index = torch.where(inputs["input_ids"][0] == tokenizer.mask_token_id)[0]

    # предсказания
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    softmax_scores = F.softmax(logits, dim=-1)

    mask_word_scores = softmax_scores[0, mask_token_index, :]

    top_k = torch.topk(mask_word_scores, top_k, dim=1)

    predicted_tokens = []
    for i in range(top_k.indices.shape[1]):
        token_id = top_k.indices[0, i].item()
        score = top_k.values[0, i].item()
        word = tokenizer.decode([token_id])
        predicted_tokens.append((word, score))

    return predicted_tokens



sentence = ["И тогда ему [MASK] утром ясно, что решение было ошибочным.",],

predictions = predict_masked_words(sentence, top_k=10)

print(f"\nПредложение: {sentence}")
print("Топ-10 предсказаний:")
for i, (word, score) in enumerate(predictions, 1):
    print(f"{i}. {word} (вероятность: {score:.4f})")




