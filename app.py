import streamlit as st

from data import (
    get_embeddings,
    load_config_data,
    get_transcripts,
    only_most_similar_embeddings,
    text_to_sentences,
)
from assistant import ask_claude
from prompt import create_prompt
from postprocess import extract_json, check_evidence
from itertools import chain

from streamlit.logger import get_logger
logger = get_logger(__name__)
logger.info('Start web app')

##### Data Load
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

logger.info('Data Loaded')
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


@st.cache_data
def get_all_sentence_embeddings(transcript_selection):
    sentences_ts = list(
            chain(
                *(get_transcripts(d) for d in transcript_selection)
        )
    )
    sentences = [s.text for s in sentences_ts]
    return sentences_ts, sentences, get_embeddings(sentences)

##### Prompt setup
# Only do request after submission 
if submit_button:
    logger.info('Submit pressed')
    # Combine multiple transcripts to get sentences (with timestamps)
    

    logger.info('Got all sentences')
    # Only picking the most similar transcripts
    # transcript = only_most_similar(user_prompt, sentences,)
    sentences_ts, sentences, all_embeddings = get_all_sentence_embeddings(transcript_selection)
    sentence_pos = only_most_similar_embeddings(user_prompt, all_embeddings)
    most_similar_sentences = [sentences[p] for p in sentence_pos]
    most_similar_sentences_ts = [sentences_ts[p] for p in sentence_pos]
    logger.info('Selected similar subset of sentences')

    transcript = ' '.join(most_similar_sentences)
    logger.info('Selected similar subset of sentences')

    st.write(f'Using your prompt:\n```{user_prompt}```')
    prompt_user_input = create_prompt(
        user_input=user_prompt,
        transcript=transcript,
        series_name=series_chosen,
    )
    logger.info('Prompt created')
    # Response via API call
    response = ask_claude(
        prompt=prompt_user_input,
        max_tokens=max_tokens,
        model_version='claude-instant-v1.1-100k',
    )
    # TODO: log information about response
    logger.info('Response from Claude completed')
    logger.info(f'Response {response["log_id"]=}')
    logger.info(f'Response {response["exception"]=}')
    logger.info(f'Response {response["stop_reason"]=}')
    logger.info(f'Response {response["stop"]=}')
    logger.info(f'Response {response["truncated"]=}')

    response_text = response['completion'].strip()
    st.write(f'**Assitant says**:\n\n{response_text}')
    # Display "equivalent JSON"
    st.write('-'*80)
    response_as_json = extract_json(
        response_text,
        max_tokens=max_tokens,
        model_version='claude-instant-v1.1',
    )
    st.json(response_as_json)
    logger.info('JSON extracted')

    st.write('### Links to Evidence')
    evidence_sentences = list(
        chain(
            *(
                text_to_sentences(ss.lower())
                for kp in response_as_json['key_points']
                for ss in kp['evidence']
            )
        )
    )
    transcript_sentences = [s.lower() for s in sentences]
    logger.info('Evidence sentences extracted')
    # TODO: Check only subset of sentences (via similarity)
    evidence_pos = check_evidence(evidence_sentences, all_embeddings[sentence_pos])
    logger.info('Checked evidence against transcripts')
    logger.info(f'{evidence_pos=}')
    for i,pos in enumerate(evidence_pos):
        if pos is not None:
            sentence = most_similar_sentences_ts[pos]
            youtube_url = sentence.source_url
            if youtube_url:
                short_url = f"http://youtu.be/{youtube_url.split('?v=')[-1]}"
                st.write(
                    f'[{sentence.text}]({short_url}?t={sentence.ts.start:.0f})'
                )

    logger.info('Script completed')