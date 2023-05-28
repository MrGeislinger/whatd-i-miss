import streamlit as st
import anthropic

from data import load_transcription
from assistant import ask_claude

#####
transcript = load_transcription('hi-018-whisper')
base_prompt = f'''{anthropic.HUMAN_PROMPT}:
Here is the full transcript of the Hello Internet podcast:
```
{transcript}
```
--------------------'''
question = 'What do the hosts from the podcast think about copyright?'
MY_PROMPT = f'{base_prompt}{question}\n{anthropic.AI_PROMPT}:'



# Use form to get user prompt & other settings
with st.form(key='user_input'):
    st.write('## Your Question')
    user_prompt = st.text_area(label='user_prompt', )
    st.write('Max tokens for output:')
    max_tokens = st.number_input(
        label='max_tokens',
        min_value=50,
        max_value=2_000,
        value=300,
        step=50,
    )
    submit_button = st.form_submit_button(label='Submit')

# Only do request after submission 
if submit_button:
    st.write(f'Using your prompt:\n```{user_prompt}```')
    prompt_user_input = (
        f'{base_prompt}\n'
        'Based on the podcast transcript above, address the following:\n'
        f'"""\n{user_prompt}\n"""\n'
        f'{anthropic.AI_PROMPT}:'
    )

    # Response via API call
    response = ask_claude(
        prompt=prompt_user_input,
        max_tokens=max_tokens,
    )
    # # TODO: log information about response
    for k in response.keys():
        if k != 'completion':
            print(f'{k}={response[k]}')
    response_text = response['completion'].strip()
    st.write(f'**Assitant says**:\n\n{response_text}')
