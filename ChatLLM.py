
from transformers import AutoModelWithLMHead, AutoTokenizer, pipeline
from langchain_core.prompts import (
    PromptTemplate
)
from langchain_huggingface.llms import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.memory import ConversationSummaryBufferMemory, ChatMessageHistory
from constanst import *
import mesop.labs as mel



def generate_answer(input: str, history: list[mel.ChatMessage]):

    template = """You are a nice chatbot having a conversation with a human.

    Previous conversation:
    {chat_history}

    New human question: {question}
    Response:"""
    prompt = PromptTemplate.from_template(template)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)
    model = AutoModelWithLMHead.from_pretrained(MODEL_CHECKPOINT, output_attentions=True)

    pipe = pipeline("text2text-generation",  model=model, tokenizer=tokenizer, max_new_tokens=512)
    llm = HuggingFacePipeline(pipeline=pipe) 
    memory_history = ChatMessageHistory()
    if len(history)>2:
        for chat in history:
            if chat.role == 'user':
                memory_history.add_user_message(chat.content)
            elif chat.role == 'assistant':
                memory_history.add_ai_message(chat.content)

    memory = ConversationSummaryBufferMemory(llm=llm, memory_key="chat_history", return_messages=True, chat_memory=memory_history)
    
    

    conversation = LLMChain(
                    llm=llm,
                    prompt=prompt,
                    verbose=True,
                    memory=memory
                )
    
    response = conversation({"question":input})    
    print(response)
    return response['text'].replace('AI:', '')


if __name__ == "__main__":
    generate_answer("teste", [])
