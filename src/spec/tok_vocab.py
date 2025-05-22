MAIN_SPECIAL_TOKENS = {
    # Message structure
    "ROLE": "<|role|>", # could also be called "<|message|>"
    "CONTENT": "<|content|>", # denotes the beginning of the message content
    "END_MESSAGE": "<|end_message|>", # denotes the end of a message
    "END_TURN": "<|end_turn|>", # denotes the end of a turn (after one or more `<|model|>` messages)
    "FORMAT": "<|format|>", # denotes the format of the message (e.g. markdown, html, etc.)

    # Special tokens
    "ESCAPE_TOKEN": "<|escape|>", # special token for escaping other special tokens
    "WRAP_TOKEN": "<|wrap|>", # a special wrapping token for nested data in messages
    "UNWRAP_TOKEN": "<|unwrap|>", # a special unwrapping token for nested data in messages

    # Message identities / speaker identities
    "MODEL_TOKEN": "<|model|>", # denotes the model or "assistant"
    "USER_TOKEN": "<|user|>", # denotes the user or "human"
    "DEVELOPER_TOKEN": "<|developer|>", # denotes the developer or "system"
    "PLATFORM_TOKEN": "<|platform|>", # denotes the platform (the AI inference service provider or host of the model)
    "CONTEXT_TOKEN": "<|context|>" # denotes "information" type messages that are to be treated as information rather than instructions
}



NUM_MEM_TOKENS = 10
MEM_TOKENS = [f"<|mem_{i}|>" for i in range(1, NUM_MEM_TOKENS + 1)]
MEM_PREFIX = "".join(MEM_TOKENS)

SPECIAL_TOKENS = {
    **MAIN_SPECIAL_TOKENS,
    **{f"MEM_{i}": token for i, token in enumerate(MEM_TOKENS, start=1)}
}

NEW_TOKENS = list(SPECIAL_TOKENS.values())

def apply_special_tokens(model, tokenizer):
    # tokenizer.add_special_tokens(
    #     special_tokens_dict={
    #         "eos_token": SPECIAL_TOKENS["END_TURN"],
    #     },
    #     replace_additional_special_tokens=True,
    # )
    
    # Add new tokens to the tokenizer
    num_added = tokenizer.add_special_tokens({'additional_special_tokens': NEW_TOKENS})
    print(f"Added {num_added} new tokens to the tokenizer")


    # Resize model embeddings to accommodate new tokens
    model.resize_token_embeddings(len(tokenizer))
    print(f"Resized model embeddings to {len(tokenizer)}")

def is_mem_token(token: str) -> bool:
    return token.startswith("<|mem_") and token.endswith("|>")