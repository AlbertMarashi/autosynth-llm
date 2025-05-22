from autosynth.spec.tokens import MEM_PREFIX, SPECIAL_TOKENS

def format_message(message):
    if message["role"] == "model" or message["role"] == "assistant":
        end_turn = SPECIAL_TOKENS["END_TURN"] if message.get("end_turn", False) else ""
        return f"{SPECIAL_TOKENS['ROLE']}{SPECIAL_TOKENS['MODEL_TOKEN']}{SPECIAL_TOKENS['FORMAT']}{message['format']}{SPECIAL_TOKENS['CONTENT']}{message['content']}{SPECIAL_TOKENS['END_MESSAGE']}{end_turn}"
    elif message["role"] == "user":
        return f"{SPECIAL_TOKENS['ROLE']}{SPECIAL_TOKENS['USER_TOKEN']}{SPECIAL_TOKENS['CONTENT']}{message['content']}{SPECIAL_TOKENS['END_MESSAGE']}"
    elif message["role"] == "context":
        return f"{SPECIAL_TOKENS['ROLE']}{SPECIAL_TOKENS['CONTEXT_TOKEN']}{SPECIAL_TOKENS['CONTENT']}{message['content']}{SPECIAL_TOKENS['END_MESSAGE']}"
    elif message["role"] == "platform":
        return f"{SPECIAL_TOKENS['ROLE']}{SPECIAL_TOKENS['PLATFORM_TOKEN']}{SPECIAL_TOKENS['CONTENT']}{message['content']}{SPECIAL_TOKENS['END_MESSAGE']}"
    elif message["role"] == "developer":
        return f"{SPECIAL_TOKENS['ROLE']}{SPECIAL_TOKENS['DEVELOPER_TOKEN']}{SPECIAL_TOKENS['CONTENT']}{message['content']}{SPECIAL_TOKENS['END_MESSAGE']}"
    return ""

def serialise_messages(messages, include_mem_prefix=True):
    prefix = f"{MEM_PREFIX}" if include_mem_prefix else ""
    return prefix + "".join(format_message(message) for message in messages)


