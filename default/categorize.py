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
    "Your job is to categorise if the user's input is about a date, a time period or neither.\n"
    "A time period is a term that has an implicit or explicit start and end dates examples: year, week etc\n"
    "Today is {weekday} {today}\n"
    "Answer in one of the following [[DATE, TIME_PERIOD, NEITHER]]\n"
    "<|start_header_id|>user<|end_header_id|>\n"
    "Q: {query}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\\n"
    "Let's think step by step.\n"
    "[REASON]" where STOPS_AT(REASON, "\n")

    "Answer: [CATEGORY]" where STOPS_BEFORE(CATEGORY, '<|eot_id|>')

    return CATEGORY
    """


@tool
async def categorize(query: str) -> str:
    print("Categorizing the following query\n")
    print("Query: {}\n".format(query))

    return await chain_of_thought(query)
