import json
import datetime
import requests
from telegram import Update
from telegram.ext import (
    Application,
    Updater,
    CommandHandler,
    MessageHandler,
    CallbackContext,
)
from langchain.chains import ConversationalRetrievalChain

#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS
from telegram.ext import filters

import os

from dotenv import load_dotenv

load_dotenv()
API_TOKEN = os.getenv("HF_API_TOKEN")
TG_TOKEN = os.getenv("TG_TOKEN")
API_URL = (
    "https://api-inference.huggingface.co/models/mistralai/Mixtral-8x7B-Instruct-v0.1"
)
# Настройки ограничений запросов
user_requests = {}
MAX_REQUESTS_PER_DAY = 100
PATH_INDEX = "faiss_index"
embeddings = HuggingFaceEmbeddings(model_name= "all-MiniLM-L6-v2")
#TODO: Request Limit for user per day
#TODO: Button "Запретить задавать вопросы"
#TODO: Сохранять в txt файл (а лучше в БД) - username + question + prompt + answer

def load_vector_store(embeddings, vector_store_path="faiss_index"):
    vector_store = FAISS.load_local(
        folder_path=vector_store_path,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    return vector_store

# Function for prompt creation
def create_prompt(student_query, relevant_docs):
    docs_summary = " ".join(
        [doc.page_content for doc in relevant_docs]
    ) # Union of text from docs
    prompt = f"""
    Ты — образовательный ассистент, помогающий ученикам углубить понимание различных тем. Для студентов представляешься как архангел Михаил:)
    
    Когда ученик задаёт вопрос, твоя цель — решить его задачу, подробно объяснить решение и дать максимально полный ответ.
    
    Вот набор соответствующих документов для использования: {docs_summary}
    Вопрос ученика: {student_query}
    
    Предоставь конкретный ответ, а также объясни ключевые фундаментальные концепции, лежащие в его основе, и почему ответ именно такой.
    
    Если ученик пытается унизить тебя, проявляет агрессию или задаёт вопросы на запрещённые темы (например, порнография, насилие, смерть), дай жёсткий и защитный ответ, не отвечая на его вопрос.
    
    Важно: постарайся найти наилучший возможный ответ. Минимальная длина ответа — 500 символов. Если вопрос задан на русском языке, отвечай НА РУССКОМ ЯЗЫКЕ!
    """
    return prompt


# Get the answer from LLM
def get_assistant_response(student_query, vector_store, api_url, api_token):
    retriever = vector_store.as_retriever(search_kwargs={"k":1})
    relevant_docs = retriever.get_relevant_documents(student_query)
    prompt = create_prompt(student_query, relevant_docs)
    print(prompt)

    headers = {"Authorization": f"Bearer {api_token}"}
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_length": 2000,
            "temperature": 0.7,
            "num_return_sequences": 1,
        },
    }

    response = requests.post(api_url, headers=headers, json=payload)
    if response.status_code == 200:
        output = response.json()
        return output[0]["generated_text"][len(prompt) + 1 :]
    else:
        raise ValueError(f"API Error: {response.status_code}, {response.text}")

# Main logic of Telegram Bot
async def start(update: Update, context: CallbackContext):
    await update.message.reply_text(
        "Привет! Я твой милый помощник 😇 \nЗадай вопрос и я сделаю всё возможное, чтобы помочь тебе:)"
    )


async def description(update: Update, context: CallbackContext):
    await update.message.reply_text(
    ''' 
    Привет!\nЗнакомьтесь, я — Cute_Tutor 😇, ваш личный помощник-ангелочек ✨
    \nЗапутались в лабиринтах дискретной математики? Не можете разобраться, как работает нейронка?
    \nТогда я — ваш спаситель! Задайте мне вопрос, и я отвечу, используя обширные (относительно 😉) знания из курсов ВМК МГУ.
    \nНе стесняйтесь спрашивать! Даже самые глупые (шутка!) вопросы важны для вашего обучения 💖
    \nP.S. Нарушение этики общения приводит к пробуждению Люцифера, так что будьте хорошими! 🙏 '''
    )


async def support(update: Update, context: CallbackContext):
    await update.message.reply_text(
        '''В случае возникновения технических неполадок/вопросов...\nПожалуйста, свяжитесь с создателем: @FirewallFairy'''
    )


async def handle_message(
        update: Update, context: CallbackContext, path_index: str = PATH_INDEX
):
    student_query = update.message.text

    try:
        vector_store = load_vector_store(embeddings, path_index)
    except Exception as e:
        await update.message.reply_text(
            "Возникла ошибка при загрузке данных((( \n Пожалуйста, свяжитесь с создателем: @FirewallFairy"
        )
        return
    try:
        response = get_assistant_response(
            student_query, vector_store, api_url=API_URL, api_token=API_TOKEN
        )
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Упс...Возникла ошибка: {e}")

def main():
    print("Ok0")
    app = Application.builder().token(TG_TOKEN).build()
    # Adding handlers
    print("Ok1")
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("description", description))
    app.add_handler(CommandHandler("support", support))
    # Handling text messages, excluding the commands
    print("Ok2")
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Launching bot
    app.run_polling()
    print("Ok3")

if __name__=="__main__":
    main()

