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
from postprocess import (
    extract_json,
    check_evidence,
    get_time_stamp_link,
    write_youtube_embed,
)
from itertools import chain, groupby
import numpy as np

##### Logging
from streamlit.logger import get_logger
logger = get_logger(__name__)
logger.info('Start web app')

#####
SENTENCE_SEPARATOR = '\n'
SECTION_SEPARATOR_START = '<section>\n'
SECTION_SEPARATOR_END = '\n</section>'

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

# data_reference = load_data('config-test.json')
data_reference = load_data('config-all.json')
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
    st.write('## DEBUG')
    debug_opt = st.checkbox(label='Display debug output', value=False)
    n_sentences = st.slider(
        label='Number of Sentences',
        min_value=30,
        max_value=500,
        step=5,
        value=200
    )
    n_buffer_before = st.slider(
        label='Buffer Sentences (before sentence)',
        min_value=0,
        max_value=50,
        step=1,
        value=3, 
    )
    n_buffer_after = st.slider(
        label='Buffer Sentences (after sentence)',
        min_value=0,
        max_value=50,
        step=1,
        value=10, 
    )
    submit_button = st.form_submit_button(label='Submit')


@st.cache_data
def get_sentence_embedding(data_info):
    sentences_ts = get_transcripts(data_info)
    sentences = [s.text for s in sentences_ts]
    identifier = data_info.get('id')
    sentence_embeddings = get_embeddings(sentences, identifier=identifier)

    return (
        sentences_ts,
        sentences,
        sentence_embeddings,
    )
    

def get_all_sentence_embeddings(transcript_selection):
    all_sentences_ts = {}
    all_sentences = {}
    all_sentence_embeddings = {}
    # TODO: More elegant way of for all the sentences in order with embeddings
    # TEMP
    s_ts = []
    s = []
    ## TEMP
    for d in transcript_selection:
        sentences_ts, sentences, sentence_embeddings = get_sentence_embedding(d)
        identifier = d.get('id')
        all_sentences_ts[identifier] = sentences_ts
        all_sentences[identifier] = sentences
        all_sentence_embeddings[identifier] = sentence_embeddings
        ## TEMP
        s_ts.extend(sentences_ts)
        s.extend(sentences)
        ## TEMP

    print(list(all_sentence_embeddings.values())[0].shape)
    final_embeddings = np.vstack(all_sentence_embeddings.values())
    print(final_embeddings.shape)

    return (
        s_ts,
        s,
        final_embeddings,
    )

##### Prompt setup
# Only do request after submission 
if submit_button:
    logger.info('Submit pressed')
    # Combine multiple transcripts to get sentences (with timestamps)
    

    logger.info('Got all sentences')
    # Only picking the most similar transcripts
    sentences_ts, sentences, all_embeddings = get_all_sentence_embeddings(transcript_selection)
    sentence_pos = only_most_similar_embeddings(
        question=user_prompt,
        embeddings=all_embeddings,
        n_sentences=n_sentences,
        n_buffer_before=n_buffer_before,
        n_buffer_after=n_buffer_after,
    )
    # Break continuous positions into "sections"
    transcript = ''
    # Group based on sequentional positions
    for _, g in groupby(enumerate(sentence_pos), (lambda ix: ix[0]-ix[1])):
        group_positions = (i for _,i in g)
        grouped_sentence = SENTENCE_SEPARATOR.join(
            sentences[p] for p in group_positions
        )
        transcript += (
            f'{SECTION_SEPARATOR_START}'
            f'{grouped_sentence}'
            f'{SECTION_SEPARATOR_END}'
        )

    most_similar_sentences_ts = [sentences_ts[p] for p in sentence_pos]
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
    # Log information about response
    logger.info('Response from Claude completed')
    logger.info(f'Response {response["log_id"]=}')
    logger.info(f'Response {response["exception"]=}')
    logger.info(f'Response {response["stop_reason"]=}')
    logger.info(f'Response {response["stop"]=}')
    logger.info(f'Response {response["truncated"]=}')

    response_text = response['completion'].strip()
    response_as_json = extract_json(
        response_text,
        max_tokens=max_tokens,
        model_version='claude-instant-v1.1',
    )
    logger.info('JSON extracted')

    # TODO: Check if data were supplied, give some output to user to try again
    # Overall Summary
    if overall_summary := response_as_json.get('overall_summary'):
        st.write('# Overall Summary')
        st.write(f'> **{overall_summary.strip()}**')
    # Each key point
    for kp in response_as_json.get('key_points', []):
        if kp_text := kp.get('text'):
            st.write('## Key Point')
            st.write(f'> {kp_text.strip()}')
            # Evidence for each key point
            evidence_sentences = list(
                chain(
                    *(
                        text_to_sentences(ss.lower())
                        for ss in kp.get('evidence', [])
                    )
                )
            )

            logger.info('Evidence sentences extracted')
            evidence_pos = check_evidence(
                evidence_sentences,
                all_embeddings[sentence_pos],
                similarity_thresh=0.7,
            )
            logger.info('Checked evidence against transcripts')
            logger.info(f'{evidence_pos=}')
            st.write('### Quotes & Timestamped Links')
            for i,pos in enumerate(evidence_pos):
                if pos is not None:
                    sentence = most_similar_sentences_ts[pos]
                    youtube_url = sentence.source_url
                    if youtube_url:
                        video_id = youtube_url.split('?v=')[-1]
                        time_start = int(sentence.ts.start)
                        short_url = get_time_stamp_link(
                            video_id=video_id,
                            time=time_start,
                        )
                        write_youtube_embed(
                            video_id=video_id,
                            time=time_start)
                        st.write(
                            f'>*{sentence.text.strip()}...*\n'
                            f'{short_url}'
                        )

    #
    if debug_opt:
        debug_section = st.expander("# DEBUG")
        # Display "equivalent JSON"
        debug_section.write(f'## Derived JSON from Assistant Output')
        debug_section.json(response_as_json, expanded=False)
        # Display assistant output
        debug_section.write(f'## Raw Output from Assistant')
        debug_section.write(f'**Assitant says**:')
        debug_section.text(f'{response_text}')
        debug_section.write('-'*80)

        logger.info('Script completed')