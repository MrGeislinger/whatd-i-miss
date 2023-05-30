import streamlit as st

from data import load_config_data, transcript_with_timestamps, only_most_similar
from assistant import ask_claude
from prompt import create_prompt

##### Data Load
# transcript = load_transcription('hi-018-whisper') # Load from local file
@st.cache_resource
def load_data(data_file: str):
    return load_config_data(data_file)

@st.cache_data
def get_episode_choices(d_ref):
    return tuple(transcript_data for transcript_data in d_ref)

@st.cache_data
def get_series_names(choices):
    different_series = set(data['series_name'] for data in choices)
    return different_series

data_reference = load_data('config.json')
episode_choices = get_episode_choices(data_reference['data'])
series_chosen = st.selectbox(
    label='Which Series?',
    options=get_series_names(episode_choices),
)
select_all_episodes = st.checkbox("Select all transcripts?")

##### User Input
# Use form to get user prompt & other settings
def get_ui_transcript_selection(select_all: bool = False):
    return container.multiselect(
        label='Which episode transcript to seach?',
        options=episode_choices,
        default=episode_choices if select_all else None,
        format_func=lambda d: d.get('episode_name'),
    )

with st.form(key='user_input'):
    # Only display choices for the given series
    episode_choices = tuple(
        e for e in episode_choices
        if e['series_name'] == series_chosen
    )
    container = st.container()
    transcript_selection = get_ui_transcript_selection(select_all_episodes)

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
# Only do request after submission 
if submit_button:
    # Combine multiple transcripts
    transcript = '\n========\n'.join(
        transcript_with_timestamps(d['url'])
        for d in transcript_selection
    )
    # Only picking the most similar transcripts
    transcript = only_most_similar(user_prompt, transcript)

    st.write(f'Using your prompt:\n```{user_prompt}```')
    prompt_user_input = create_prompt(
        user_input=user_prompt,
        transcript=transcript,
        series_name=series_chosen,
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
