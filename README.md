<!-- О Проекте -->
## О проекте


Telegram чат-бот на основе `RAG` - подхода. <br/>
Разрабатывался в осеннюю сессию для помощи при подготовке к пересдаче. <br/>
Подход заключается в создании локальной векторной бд (`faiss`),
из которой находится relevant info по промпту от пользователя, внутри векторной бд происходит сравнение embeddings для поиска необходимой информации. <br/>
Далее в llm модель (`BeRT`) 
подаётся промт, содержащий тип поведения (в данном случае - `Милый помощник`, prompt от пользователя, негативный промп, а также указание использовать relevant info). <br/>
Сгенерированный ответ отображается в чате. <br/>
Для настройки требуется создание бота в Telegram в FatherBot, полученный API ключ требуется указать в файле .env.
В репозитории также находятся файлы - `controlling_questions.pdf` и `exam_questions.pdf`, папка `faiss` - уже собранная бд на основании приведённых .pdf.
<p align="right">(<a href="#readme-top">back to top</a>)</p>

### Технологии

* [Python](https://www.python.org/)
* [Jupyter](https://jupyter.org/)
* [Faiss](https://ai.meta.com/tools/faiss/)
* [HuggingFace](https://huggingface.co/)
* [RAG-подход](https://cloud.google.com/use-cases/retrieval-augmented-generation)

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Фотографии собранной и настроенной версии
/start <br/>  <br/>
<img width="633" alt="start" src="https://github.com/user-attachments/assets/4fa31a45-cc03-4c72-9c65-153267e04b2a" /> 
<br/>  <br/>
/functions <br/>  <br/>
<img width="629" alt="functions" src="https://github.com/user-attachments/assets/2eb7e96b-9008-4eea-96ff-9351b8e26c2c" />
<br/>  <br/>
/description <br/>  <br/>
<img width="635" alt="description" src="https://github.com/user-attachments/assets/ae30663b-a015-413f-827c-1366967047aa" />
<br/>  <br/>
/question <br/>  <br/>
<img width="633" alt="question" src="https://github.com/user-attachments/assets/ed464b51-3199-4a75-9666-98497df83ef2" />
<br/>  <br/>


