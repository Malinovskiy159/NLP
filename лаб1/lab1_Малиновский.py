import pymorphy3
import nltk
from nltk.tokenize import word_tokenize
from nltk import sent_tokenize
nltk.download('punkt')

morph = pymorphy3.MorphAnalyzer()

def segmentetion(text):
    return sent_tokenize(text)



def russian_tokenize(text):
    return word_tokenize(text)
    #return re.findall(r'[а-яёА-ЯЁ-]+|[.,!?;:]', text)


def is_adjective(parsed_word):
    if not parsed_word.tag.POS:
        return False
    return parsed_word.tag.POS in {'ADJF', 'ADJS'}

def is_noun(parsed_word):
    if not parsed_word.tag.POS:
        return False

    return parsed_word.tag.POS == 'NOUN'


def filter_plausible_parses(parses, expected_pos=None):
    plausible_parses = []

    for parse in parses:
        # пропускаем разборы с низкой вероятностью
        if parse.score < 0.01:
            continue

        """# пропускаем устаревшие и редкие формы
        if any(mark in str(parse.tag) for mark in ['Arch', 'Rare', 'Slang', 'Erro']):
            continue


        # Дополнительные фильтры для конкретных частей речи
        if is_adjective(parse):
            # Для прилагательных пропускаем редкие падежи
            if parse.tag.case in ['voct', 'loc2', 'gen2', 'acc2']:
                continue

        if is_noun(parse):
            # Для существительных пропускаем звательный падеж и др.
            if parse.tag.case in ['voct', 'loc2']:
                continue
"""
        plausible_parses.append(parse)

    # Сортируем по вероятности (score)
    #plausible_parses.sort(key=lambda x: x.score, reverse=True)

    return plausible_parses


def has_grammatical_agreement(adj_parse, noun_parse):

    # Обязательное совпадение падежа
    if not (adj_parse.tag.case and noun_parse.tag.case and
            adj_parse.tag.case == noun_parse.tag.case):
        return False

    # Обязательное совпадение числа
    if not (adj_parse.tag.number and noun_parse.tag.number and
            adj_parse.tag.number == noun_parse.tag.number):
        return False

    # Для единственного числа проверяем род
    if (adj_parse.tag.number == 'sing' and
            adj_parse.tag.gender and noun_parse.tag.gender and
            adj_parse.tag.gender != noun_parse.tag.gender):
        return False


    return True


def find_plausible_matching_pair(adj_word, noun_word):

    # Получаем и фильтруем разборы
    adj_parses = filter_plausible_parses(morph.parse(adj_word), is_adjective)
    noun_parses = filter_plausible_parses(morph.parse(noun_word), is_noun)

    if not adj_parses or not noun_parses:
        return None

    # самые вероятные варианты (топ-3)
    top_adj_parses = adj_parses[:3]
    top_noun_parses = noun_parses[:3]

    # комбинации самых вероятных вариантов
    for adj_parse in top_adj_parses:
        for noun_parse in top_noun_parses:
            if has_grammatical_agreement(adj_parse, noun_parse):
                return (adj_parse.normal_form, noun_parse.normal_form)

    return None



def find_agreed_pairs_strict(tokens):
    pairs = []

    for i in range(len(tokens) - 1):
        word1 = tokens[i].lower()
        word2 = tokens[i + 1].lower()

        # короткие слова и знаки препинания
        if (len(word1) <= 1 or len(word2) <= 1 or
                word1 in '.,!?;:()[]{}"\'«»' or
                word2 in '.,!?;:()[]{}"\'«»'):
            continue

        # отбор пар
        result = find_plausible_matching_pair(word1, word2)
        if result:
            pairs.append(result)
            continue

    return pairs


# программа
try:
    with open('input.txt', 'r', encoding='utf-8') as file:
        text = file.read()
except UnicodeDecodeError:
    try:
        with open('input.txt', 'r', encoding='windows-1251') as file:
            text = file.read()
    except UnicodeDecodeError:
        print("Не удалось прочитать файл. Проверьте кодировку.")
        exit()

segments=segmentetion(text)
print(segments)

tokens = russian_tokenize(text)
print(tokens)

agreed_pairs = find_agreed_pairs_strict(tokens)
print(agreed_pairs)

print("Найдены правдоподобные согласованные пары:")
for adj, noun in agreed_pairs:
    print(f"{adj} {noun}")

print(f"\nВсего найдено пар: {len(agreed_pairs)}")