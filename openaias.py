from openai1 import OpenAI
client = OpenAI(base_url="http:localhost:1234/v1", api_key="not-needed")
relevant_context = get_relevant_context(user_input, vault_embeddings, vault_context, model)
# print(relevant_context) #debugging
user_input_with_context = user_input
if relevant_context:
    user_input_with_context = "\n".join(relevant_context) + "\n\n" + user_input
messages = [{"role": "system", "content": system_message}] + conversation_history + [{"role": "user", "content": user_input_with_context}]
temperature = 1
streamed_completion = client.chat.completions.create(
model="local-model",
messages=messages,
stream=True)
full_response = ""
line_buffer = ""
for chunk in streamed_completion:
delta_content = chunk.choices[0].delta.content
if delta_content:
line_buffer += delta_content
if '\n' in line_buffer:
lines = line_buffer.split('\n')
for line in lines[:-1]:
print(NEON_GREEN + line + RESET_COLOR)
full_response += line + "\n"
line_buffer = lines[-1]
if line_buffer:
print(NEON_GREEN + line_buffer + RESET_COLOR)
full_response += line_buffer