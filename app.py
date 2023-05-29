import streamlit as st

from data import load_transcription, load_from_url, load_config_data, transcript_with_timestamps
from assistant import ask_claude
from prompt import create_prompt

##### Data Load
# transcript = load_transcription('hi-018-whisper') # Load from local file
data_reference = load_config_data('config.json')

ALL_EPISODE_KEY = 'ALL EPISODES'
episode_choices = tuple(
    [
        transcript_data
        for transcript_data in data_reference['data']
    ] +
    [ALL_EPISODE_KEY]
)

##### User Input
# Use form to get user prompt & other settings
with st.form(key='user_input'):
    transcript_option = st.selectbox(
        label='Which episode transcript to seach?',
        options=episode_choices,
        format_func=(
            lambda d: (
                d.get('episode_name') if d != ALL_EPISODE_KEY
                else ALL_EPISODE_KEY
            )
        ),
        index=0,
    )
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


##### Prompt setup
if transcript_option == ALL_EPISODE_KEY:
    transcript = '\n========\n'.join(
        transcript_with_timestamps(d['url'])
        for d in data_reference['data']
    )
else:
    transcript = transcript_with_timestamps(transcript_option['url'])


##### Response to user's question
# Only do request after submission 
if submit_button:
    st.write(f'Using your prompt:\n```{user_prompt}```')
    series_name = (
        None if transcript_option == ALL_EPISODE_KEY
        else transcript_option['series_name']
    )
    prompt_user_input = create_prompt(
        user_input=user_prompt,
        transcript=transcript,
        series_name=series_name,
    )

    # Response via API call
    response = ask_claude(
        prompt=prompt_user_input,
        max_tokens=max_tokens,
        model_version='claude-instant-v1-100k',
    )
    # TODO: log information about response
    for k in response.keys():
        if k != 'completion':
            print(f'{k}={response[k]}')

    response_text = response['completion'].strip()
    st.write(f'**Assitant says**:\n\n{response_text}')
