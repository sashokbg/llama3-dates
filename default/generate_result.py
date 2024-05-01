# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from datetime import datetime

import lmql
from promptflow.core import tool

model = lmql.model("llama.cpp:/home/alexander/Games2/models/Meta-Llama-3-8B-Instruct.Q5_K_M.gguf",
                   tokenizer="meta-llama/Meta-Llama-3-8B", endpoint="localhost:9999")

today = datetime.today().strftime('%d-%m-%Y')
weekday = datetime.now().strftime('%A')


def validate_date(date_str: str):
    if " - " in date_str:
        date1 = datetime.strptime(date_str.split(" - ")[0], '%d-%m-%Y').date()
        date2 = datetime.strptime(date_str.split(" - ")[1], '%d-%m-%Y').date()
        return "{} {} - {} {}".format(date1.strftime('%A'), date1, date2.strftime('%A'), date2)
    else:
        date = datetime.strptime(date_str, '%d-%m-%Y').date()
        return "{} {}".format(date.strftime('%A'), date)


# @lmql.query(decoder="argmax", model=model, output_writer=lmql.printing)
@lmql.query(decoder="argmax", model=model, output_writer=lmql.printing)
async def chain_of_thought(query, category) -> str:
    """lmql
    "<|begin_of_text|>\n"
    "<|start_header_id|>system<|end_header_id|>\n"
    "You are a helpful assistant that translates human input to dates.\n"
    "Expected output is either a single date DD-MM-YYYY or a date range DD-MM-YYYY - DD-MM-YYYY\n"
    "The user input has been categorized as a {category}\n"
    "Invalid dates are marked as INVALID\n"
    "Today is {weekday} {today} (DD-MM-YYYY)\n"
    "<|start_header_id|>user<|end_header_id|>Q: {query}<|eot_id|>\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "Let's think step by step.\n"
    "[REASON]" where STOPS_AT(REASON, "\n\n")
    "Therefore the answer is: [DATE]\n" where STOPS_BEFORE(DATE, "<|eot_id|>") or STOPS_AT(DATE, "\n\n")
    "Let's see what the calendar says: \n"
    "<|start_header_id|>calendar<|end_header_id|>\n"
    "{validate_date(DATE.strip())}\n"
    "<|start_header_id|>assistant<|end_header_id|>\n"
    "So final answer is:  [FINAL_DATE]" where STOPS_BEFORE(FINAL_DATE, "<|eot_id|>") or FINAL_DATE(DATE, "\n\n")
    return FINAL_DATE.strip() + ''
    """


@tool
async def generate_result(query: str, category: str) -> str:
    return await chain_of_thought(query, category)
