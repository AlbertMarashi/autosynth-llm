
conversations = [
    [
        {
            "content": "You are a stock price assistant. You can use the `stock_price:json` tool to get current prices.\n\nAlways use `reasoning` format first to plan your response, then call the tool, and provide results in `markdown`.",
            "role": "developer"
        },
        {
            "content": "What's the current price of AAPL?",
            "role": "user"
        },
        {
            "content": "I need to check the current price of Apple stock (AAPL) using the stock_price tool, then present the information to the user.",
            "end_turn": False,
            "format": "reasoning",
            "role": "model"
        },
        {
            "content": "{\"symbol\": \"AAPL\"}",
            "end_turn": False,
            "format": "stock_price:json",
            "role": "model"
        },
        {
            "content": "{\"symbol\": \"AAPL\", \"price\": 187.63, \"change\": 1.28, \"percent_change\": 0.69, \"last_updated\": \"2025-07-15T16:00:00Z\"}",
            "name": "stock_price:json",
            "role": "context"
        },
        {
            "content": "# Apple Inc. (AAPL)\n\nCurrent stock price: **$187.63**\n\nChange: +$1.28 (**+0.69%**)\n\nLast updated: July 15, 2025 at 16:00 UTC",
            "end_turn": True,
            "format": "markdown",
            "role": "model"
        }
    ],
    [
        {
            "content": "You are a stock price assistant. You can use the `stock_price:json` tool to get current prices.\n\nAlways use `reasoning` format first to plan your response, then call the tool, and provide results in `markdown`.",
            "role": "developer"
        },
        {
            "content": "What's the current price of AAPL?",
            "role": "user"
        },
        {
            "content": "I need to check the current price of Apple stock (AAPL) using the stock_price tool, then present the information to the user.",
            "end_turn": False,
            "format": "reasoning",
            "role": "model"
        },
        {
            "content": "{\"symbol\": \"AAPL\"}",
            "end_turn": False,
            "format": "stock_price:json",
            "role": "model"
        },
        {
            "content": "{\"symbol\": \"AAPL\", \"price\": 187.63, \"change\": 1.28, \"percent_change\": 0.69, \"last_updated\": \"2025-07-15T16:00:00Z\"}",
            "name": "stock_price:json",
            "role": "context"
        },
        {
            "content": "# Apple Inc. (AAPL)\n\nCurrent stock price: **$187.63**\n\nChange: +$1.28 (**+0.69%**)\n\nLast updated: July 15, 2025 at 16:00 UTC",
            "end_turn": True,
            "format": "markdown",
            "role": "model"
        }
    ]
]

# print(conversations)


# from transformers import AutoTokenizer

# tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-0324")
# num_new_tokens = tokenizer.add_special_tokens({'additional_special_tokens': list(SPECIAL_TOKENS.values())})

# # Create dataset
# dataset = CustomDataset(conversations, tokenizer, max_length=128)

# example = dataset[0]

# def display_token_table(input_ids: torch.Tensor, loss_multiplier: torch.Tensor, tokenizer: Any):
#     """
#     Display a table with columns: token id, text of that token, score.
#     """
#     print("\nToken Table")
#     print("-" * 50)
#     print(f"{'Token ID':<10} {'Text':<20} {'Score':<10}")
#     print("-" * 50)

#     for i, (token_id, score) in enumerate(zip(input_ids, loss_multiplier)):
#         # Decode single token ID to text
#         text = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
#         # Escape newlines and special characters for display
#         text = text.replace("\n", "\\n").replace("\r", "\\r")
#         print(f"{token_id.item():<10} {text:<20} {score.item():<10.2f}")

#     print("-" * 50)


# # display_token_table(example['input_ids'], example['loss_multiplier'], tokenizer)

# print(example)
