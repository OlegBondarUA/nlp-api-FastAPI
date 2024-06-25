# NLTK FastAPI REST API

Цей проект створює REST API з використанням бібліотеки NLTK для обробки тексту.

### Вимоги

Перед використанням проекту переконайтеся, що встановлено Python 3.x та усі необхідні залежності, включені до `requirements.txt`.

### Інсталяція

1. Створіть та активуйте віртуальне середовище Python:
   ```bash
   python -m venv venv
   source venv/bin/activate  # для Unix/macOS
   venv\Scripts\activate  # для Windows
   
2. Встановіть залежності з файлу requirements.txt:
   ```bash
   pip install -r requirements.txt

3. Запустіть сервер за допомогою uvicorn:
   ```bash
   uvicorn main:app --reload

Використання
API має наступні ендпоінти:

/tokenize: Токенізація тексту.
/pos_tag: Частиномовна розмітка тексту.
/ner: Розпізнавання іменованих сутностей.
Кожен з цих ендпоінтів підтримує метод POST з JSON тілом, яке містить поле text.


Приклад використання
Використайте Postman або інший інструмент для HTTP запитів, щоб взаємодіяти з API. Наприклад:

POST запит до http://127.0.0.1:8000/tokenize 
POST запит до http://127.0.0.1:8000/pos_tag
POST запит до http://127.0.0.1:8000/ner з JSON тілом:

{
    "text": "Barack Obama was born in Hawaii and worked in Washington D.C."
}
