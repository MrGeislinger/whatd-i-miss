import streamlit as st
import os

from data import (
    get_embeddings,
    load_config_data,
    get_transcripts,
    only_most_similar_embeddings,
    text_to_sentences,
    download_embeddings,
)
from assistant import ask_claude, calculate_tokens, MODELS
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
ONLINE_CONFIG_URL = 'https://github.com/MrGeislinger/whatd-i-miss/releases/download/v1.0.1/config.json'
PRECOMPUTED_EMBEDDINGS_URL = 'https://github.com/MrGeislinger/whatd-i-miss/releases/download/v1.0.2/precomputed_embeddings.tar.gz'
MAX_N_TRANSCRIPTS = 50
SENTENCE_SEPARATOR = '\n'
SECTION_SEPARATOR_START = '<section>\n'
SECTION_SEPARATOR_END = '\n</section>'

##### HEADER

st.write('# :green[W]:blue[I]:violet[M] - :green[What\'d] :blue[I] :violet[Miss]:orange[?]')
st.write(
    'Ask pointed questions about a given playlist '
    'and get back a :green[summary], :blue[key points], and related '
    ':violet[timestamps] :orange[generated via AI]! 🤖\n\n'
    'Could be podcast series, a learning series, or something completely '
    'different! Can take in even very large/long series (tested on '
    '[~150 ~2-hour long podcasts](https://www.youtube.com/playlist?list=PLe_b-HAZD1pXZl1UzE7Q9IiYMXKxSG7Lg))!'
    '\n\n'
    'Tool created for [lablab.ai\'s](https://lablab.ai/) 2023 '
    '[Anthropic AI Hackathon](https://lablab.ai/event/anthropic-ai-hackathon)'
)
st.write('-'*80)

#####
@st.cache_resource
def get_precomputed_embeddings():
    logger.info('Download embeddings')
    st.write(
        'Downloading pre-computed embeddings...\n(this could take a while...)'
    )
    download_embeddings(PRECOMPUTED_EMBEDDINGS_URL)
    st.write('Downloaded embeddings!')

logger.info('Search for embeddings')
embed_dir = os.listdir('data/_embeddings')
if len(embed_dir) <= 1:
    get_precomputed_embeddings()

##### Since I can't always use my API
user_api_key: str | None = st.text_input(
    label='Enter your Anthropic API Key',
    type='password',
    value='',
)
st.write('''
> Use your Anthropic API Key to have this tool definitely work. There's a 
> chance the default key will work but only one instance can run at a time.
''')
user_api_key = user_api_key if user_api_key else None
st.write('-'*80)
##### Data Load
@st.cache_resource
def load_data(data_file: str, online: bool = False):
    return load_config_data(data_file, online)

@st.cache_data
def get_episode_choices(d_ref):
    return tuple(transcript_data for transcript_data in d_ref)

@st.cache_data
def get_series_names(choices):
    different_series = set(data['series_name'] for data in choices)
    return different_series

# Allow online config
try:
    logger.info('Load from url?')
    data_reference = load_data(ONLINE_CONFIG_URL, online=True)
    logger.info('Loaded from url')
except:
    logger.info('Loaded from url failed; try local version')
    data_reference = load_data('config.json')

episode_choices = get_episode_choices(data_reference['data'])

st.write('# Checkout a Series...')
series_chosen = st.selectbox(
    label='Which Series?',
    options=get_series_names(episode_choices),
)
if series_chosen == 'Hello Internet':
    select_all_episodes = False
else:
    select_all_episodes = st.checkbox("Select all transcripts?")
st.write(
    '> Note: Due to limited demo resources, only up to '
    f'{MAX_N_TRANSCRIPTS} transcripts can be used'
)

logger.info('Data Loaded')
##### User Input
# Use form to get user prompt & other settings
def get_ui_transcript_selection(select_all: bool = False):
    return container.multiselect(
        label='Which episode transcript to search?',
        options=episode_choices,
        default=episode_choices if select_all else None,
        format_func=lambda d: d.get('episode_name'),
        max_selections=MAX_N_TRANSCRIPTS,
    )

with st.form(key='user_input'):
    # Only display choices for the given series
    episode_choices = tuple(
        e for e in episode_choices
        if e['series_name'] == series_chosen
    )
    container = st.container()
    transcript_selection = get_ui_transcript_selection(select_all_episodes)

    st.write('# Ask a Question or Write a Topic')
    user_prompt = st.text_area(
        label='Ask a question or state a topic of interest', 
        value='What do the speakers talk about?',
        max_chars=500,
    )

    max_tokens = st.number_input(
        label='How long should the output be at the most? (max_tokens)',
        min_value=100,
        max_value=3_000,
        value=900,
        step=50,
    )

    adv_opts = st.expander('Advanced Options')
    adv_opts.write('## DEBUG')
    debug_opt = adv_opts.checkbox(label='Display debug output', value=False)
    model_version = adv_opts.selectbox(
        label='Model Selection for Generated Summaries',
        options=MODELS.keys(),
        index=0,
    )
    n_sentences = adv_opts.slider(
        label='Number of (Similar) Sentences to use',
        min_value=30,
        max_value=500,
        step=5,
        value=200,
    )
    n_buffer_before = adv_opts.slider(
        label='Buffer Sentences (before main sentence)',
        min_value=0,
        max_value=50,
        step=1,
        value=3, 
    )
    n_buffer_after = adv_opts.slider(
        label='Buffer Sentences (after main sentence)',
        min_value=0,
        max_value=50,
        step=1,
        value=10, 
    )
    use_all_text_if_possible = adv_opts.checkbox(
        label='Use all transcripts if model allows?',
        value=True,
    )

    submit_button = st.form_submit_button(label='Submit')
    verify_button = st.form_submit_button(label='Verify')

if verify_button:
    st.write(
        'Tokens from user input: '
        f'**{calculate_tokens(user_prompt, model_version)}**'
    )

@st.cache_data(persist='disk')
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
    logger.info(f'{transcript_selection=}')
    for i, d in enumerate(transcript_selection):
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
if verify_button or submit_button:
    logger.info('Submit pressed')
    # Combine multiple transcripts to get sentences (with timestamps)
    
    logger.info('Got all sentences')
    # Only picking the most similar transcripts
    st.write(
        'Finding what to feed the bot 🤖\n'
        '(This can take a bit with many/long transcripts)'
    )
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
    # Group based on sequential positions
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
    total_tokens = calculate_tokens(prompt_user_input, model_version)
    logger.info(f'Tokens from user input: {total_tokens=}')
    st.write(
        'Total tokens to be sent '
        '(from user prompt & selected transcript sections): \n'
        f'**{total_tokens:,}**'
    )
    tokens_allowed = MODELS[model_version][0]
    # Warn user that's too many slices (tokens)
    if total_tokens > tokens_allowed:
        st.write(
            f'You are using {model_version=}\n'
            f'This model can take about :green[**{tokens_allowed:,}** tokens] '
            f'but your input will use :red[***{total_tokens:,} tokens***].\n\n'
            'This can cause output to not receive full context. Consider '
            'reducing the input by adjusting number of sentences and/or number '
            'of buffer sentences in the ["Advanced Options"](#debug) section.'
        )
    elif use_all_text_if_possible: # See if all the text can be used
        logger.info('Check if use full transcript ')
        all_sentences = SENTENCE_SEPARATOR.join(
            s_text for s_text in sentences
        )
        prompt_all_sentences = create_prompt(
            user_input=user_prompt,
            transcript=all_sentences,
            series_name=series_chosen,
        )
        all_sentences_tokens = calculate_tokens(
            prompt=prompt_all_sentences,
            model_version=model_version,
        )
        if tokens_allowed > all_sentences_tokens:
            logger.info(f'Can use all text for model! {all_sentences_tokens=}')
            st.write(
                f'Using all text from transcript since you can use '
                f':blue[{tokens_allowed:,} tokens] but using all the text is '
                f'only :green[{all_sentences_tokens:,} tokens]!'
            )
            prompt_user_input = prompt_all_sentences
    if verify_button and (not submit_button):
        st.write('You can now make a submit or make changes')

# Only do request after submission 
if submit_button:
    st.write(
        'Sending request to model (depending on model could take a few '
        'seconds to over a minute).\n\n'
        '-----------------'
    )
    # Response via API call
    response = ask_claude(
        prompt=prompt_user_input,
        max_tokens=max_tokens,
        model_version=model_version,
        api_key=user_api_key,
    )
    # Log information about response
    logger.info('Response from Claude completed')
    logger.info(f'Response {response.id=}')
    logger.info(f'Response {response.stop_reason=}')
    logger.info(f'Response {response.stop_sequence=}')

    response_text = response.content[0].text.strip()
    response_as_json = extract_json(
        response_text,
        max_tokens=max_tokens,
        model_version=model_version,
        api_key=user_api_key,
    )
    logger.info('JSON extracted')

    # TODO: Check if data were supplied, give some output to user to try again
    # Overall Summary
    if overall_summary := response_as_json.get('overall_summary'):
        st.write('# :green[Overall Summary]')
        st.write(f'> :orange[**{overall_summary.strip()}**]')
    # Each key point
    for kp in response_as_json.get('key_points', []):
        if kp_text := kp.get('text'):
            st.write('## :blue[Key Point]')
            st.write(f'> :orange[{kp_text.strip()}]')
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
            st.write('### :violet[Quotes & Timestamped Links]')
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
                        st.write(
                            f'>*"{sentence.text.strip()}..."*\n'
                            f'{short_url}'
                        )
                        write_youtube_embed(
                            video_id=video_id,
                            time=time_start)
            st.write('-'*80)
    #
    if debug_opt:
        debug_section = st.expander("# DEBUG")
        # Display "equivalent JSON"
        debug_section.write(f'## Derived JSON from Assistant Output')
        debug_section.json(response_as_json, expanded=False)
        # Display assistant output
        debug_section.write(f'## Raw Output from Assistant')
        debug_section.write(f'**Assistant says**:')
        debug_section.text(f'{response_text}')
        debug_section.write('-'*80)

        logger.info('Script completed')