import json
from assistant import attempt_claude_fix_json
from data import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity

def extract_json(
    text_with_json: str,
    max_attempts: int = 3,
    attempt: int = 0,
    **claude_kwargs,
) -> dict | None:
    # Check if reached the max number of attempts (recursive)
    if attempt <= max_attempts:
        # Assuming the only brackets contain just json
        index_start = text_with_json.find('{')
        index_final = text_with_json.rfind('}')
        potential_json = text_with_json[index_start:index_final+1].strip()
        try:
            json_out = json.loads(potential_json)
            return json_out
        except Exception as err:
            # If error, attempt to use Claude to fix the JSON
            print(f'ERROR -> {err}')
            # Check that it's not just a simple empty JSON object
            if text_with_json[index_start+1] == '}':
                return {}
            print('Attempting Claude to fix JSON....')
            # Note this can modify original data but in practice looks good...
            new_text = attempt_claude_fix_json(potential_json, **claude_kwargs)
            # Recursively call attempt to extract (valid) JSON
            return extract_json(new_text, attempt=attempt+1)
    else:
        # Never finished...
        raise Exception('ERROR -> Could not fix JSON')
    
def filter_evidence_sentence_positions(
    evidence_score_pos: list[tuple[int, float]],
    score_thresh: float = 0.75,
    n_sentences_buffer: int = 5,
) -> list[int]:
    positions = []
    # If position is within buffer send the first value (if over threshold)
    for escore, epos in evidence_score_pos:
        # Check if similar enough
        if escore >= score_thresh:
            # Position is farther away from last position by given buffer
            if len(positions) < 1:
                positions.append(epos)
            elif epos > positions[-1] + n_sentences_buffer:
                positions.append(epos)
    return positions

# Compare returned "evidence" to actual transcript
def check_evidence(
    evidence_sentences: list[str],
    transcript_embeddings,
    similarity_thresh: float = 0.75,
    n_sentences_buffer: int = 5,
) -> list[tuple[int, float]]:
    '''Use transcript sentences with timestamps
    '''
    # Find most similar sentence from evidence
    evidence_similarities = cosine_similarity(
        get_embeddings(evidence_sentences),
        transcript_embeddings,
    )
    # Get the most similar transcript sentence to evidence sentence
    sentence_positions = evidence_similarities.argmax(axis=1)
    # Positions of most related(relative to transcript sentences)
    evidence_score_pos = [
        (evidence_similarities[i, pos], sentence_positions[i])
        for i, pos in enumerate(sentence_positions)
    ]

    # Filters so links will be separated by n sentences
    evidence_positions = filter_evidence_sentence_positions(
        evidence_score_pos=evidence_score_pos,
        n_sentences_buffer=n_sentences_buffer,
        score_thresh = similarity_thresh,
    )

    # TODO: If evidence not similar enough, maybe alternatively see if evidence
    # is in a sentence from the transcript.

    return evidence_positions

