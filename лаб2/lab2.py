import nltk
import gensim
import re

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)



# ОСНОВНОЙ КОД ДЛЯ WORD2VEC
def main_word2vec():
    try:
        # Загрузка модели
        word2vec = gensim.models.KeyedVectors.load_word2vec_format("cbow.txt", binary=False)

    except Exception as e:
        print(f"Ошибка загрузки модели: {e}")
        return

    #  фильтрация существительных
    pat = re.compile("(.*)_NOUN")

    # Ваши целевые слова
    target1 = "бульвар"
    target2 = "прохожий"

    print(f"Поиск комбинаций для слов: '{target1}' и '{target2}'")
    print("=" * 60)

    # Тестовые комбинации
    test_combinations = [
        (["человек_NOUN", "пешеход_NOUN", "улица_NOUN", "сквер_NOUN"], [], "человек + пешеход + улица + сквер"),
        (["житель_NOUN", "улица_NOUN", "парк_NOUN", "аллея_NOUN"], [], "житель + улица + парк + аллея"),
        (["пешеход_NOUN", "город_NOUN", "проспект_NOUN", "тропа_NOUN"], [], "пешеход + город + проспект + тропа"),
        (["человек_NOUN", "пешеход_NOUN", "парк_NOUN", "тропа_NOUN"], [], "человек + пешеход + парк + тропа"),
        (["житель_NOUN", "улица_NOUN", "сквер_NOUN", "тропа_NOUN"], [], "житель + улица + сквер + тропа"),
        (["пешеход_NOUN", "человек_NOUN", "аллея_NOUN", "улица_NOUN"], [], "пешеход + человек + аллея + улица"),
        (["человек_NOUN", "город_NOUN", "парк_NOUN", "пешеход_NOUN"], [], "человек + город + парк + пешеход"),
    ]

    successful_combinations = []
    for pos, neg, desc in test_combinations:
        try:
            print(f"\nПробуем: {desc}")
            dist = word2vec.most_similar(positive=pos, negative=neg, topn=15)

            # Фильтруем только существительные
            noun_results = []
            for word, similarity in dist:
                match = pat.match(word)
                if match is not None:
                    clean_word = match.group(1)
                    noun_results.append((clean_word, similarity))

            # Проверяем наличие целевых слов в топ-10
            found_targets = []
            print("Топ-10 результатов:")
            for i, (word, sim) in enumerate(noun_results[:10]):
                marker = " <<< ЦЕЛЕВОЕ" if word in [target1, target2] else ""
                print(f"{i + 1:2d}. {word:<20} {sim:.4f}{marker}")
                if word in [target1, target2]:
                    found_targets.append(word)

            if len(found_targets) == 2:
                print(f" Найдены оба целевых слова ")

            elif found_targets:
                print(f" Найдено частично: {found_targets}")

        except Exception as e:
            print(f"Ошибка: {e} - возможно, слова нет в словаре")
            continue

if __name__ == "__main__":
    main_word2vec()

