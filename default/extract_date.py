from datetime import datetime

import lmql
from promptflow.core import tool

model = lmql.model("llama.cpp:/home/alexander/Games2/models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
                   tokenizer="meta-llama/Meta-Llama-3-8B", endpoint="localhost:9999")

today = datetime.today().strftime('%d-%m-%Y')
weekday = datetime.now().strftime('%A')


@lmql.query(decoder="argmax", model=model)
async def chain_of_thought(query) -> str:
    """lmql
    "<|begin_of_text|>\n"
    "<|start_header_id|>system<|end_header_id|>\n"
    "Your job is to extract the part of user's input that contains a date expression or time period.\n"
    "Your answer should only contain the date or expression\n"
    "Answer 'N/A' if the phrases does not mention any date or time period.\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Q: {query}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\\n"

    "The date expression in the sentence is: [RESULT]" where STOPS_BEFORE(RESULT, '<|eot_id|>')

    return RESULT
    """


@tool
async def extract_date(query: str) -> str:
    print("Query: {}\n".format(query))

    return await chain_of_thought(query)

