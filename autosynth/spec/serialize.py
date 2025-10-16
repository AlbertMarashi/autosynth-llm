from autosynth.spec.tokens import MEM_PREFIX, SPECIAL_TOKENS

def serialise_lson_content(content):
    if isinstance(content, str):
        return SPECIAL_TOKENS["STRING_TOKEN"] + content
    elif isinstance(content, (int, float)):
        return SPECIAL_TOKENS["NUMBER_TOKEN"] + str(content)
    elif isinstance(content, bool):
        return SPECIAL_TOKENS["BOOLEAN_TOKEN"] + str(content)
    elif content is None:
        return SPECIAL_TOKENS["NULL_TOKEN"]
    elif isinstance(content, list):
        return serialise_lson_array(content)
    elif isinstance(content, dict):
        return serialise_lson_object(content)
    else:
        raise ValueError(f"Unknown content type: {type(content)}")

def serialise_lson_array(array):
    return SPECIAL_TOKENS["ARRAY_START_TOKEN"] + "".join(serialise_lson_content(item) for item in array) + SPECIAL_TOKENS["ARRAY_END_TOKEN"]

def serialise_lson_object(obj):
    return (SPECIAL_TOKENS["OBJECT_START_TOKEN"] + "".join(SPECIAL_TOKENS["OBJECT_KEY_TOKEN"] + key + serialise_lson_content(value) for key, value in obj.items()) + SPECIAL_TOKENS["OBJECT_END_TOKEN"])


def serialise_message(message):
    role = message.get("role")
    if role == "end_turn": return SPECIAL_TOKENS["END_TURN_TOKEN"]
    # Create a copy of the message without the "role" key
    message = {k: v for k, v in message.items() if k != "role"}

    return SPECIAL_TOKENS["ROLE"] + token_for_role(role) + serialise_lson_content(message)    

def token_for_role(role):
    if role == "model": return SPECIAL_TOKENS["MODEL_TOKEN"]
    elif role == "user": return SPECIAL_TOKENS["USER_TOKEN"]
    elif role == "context": return SPECIAL_TOKENS["CONTEXT_TOKEN"]
    elif role == "developer": return SPECIAL_TOKENS["DEVELOPER_TOKEN"]
    elif role == "platform": return SPECIAL_TOKENS["PLATFORM_TOKEN"]
    elif role == "end_turn": return SPECIAL_TOKENS["END_TURN_TOKEN"]
    else: raise ValueError(f"Unknown role: {role}")

def serialise_messages(messages, include_mem_prefix=True):
    prefix = f"{MEM_PREFIX}" if include_mem_prefix else ""
    return prefix + "".join(serialise_message(message) for message in messages)


