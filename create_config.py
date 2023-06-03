from pathlib import Path
from json import dumps
from hashlib import sha256
import random


def single_config_file(fpath: Path, series_name: str) -> dict:
    file = str(fpath.relative_to('.'))
    episode_name = str(fpath.stem)
    youtube_id = episode_name.split('.')[-1]
    # This will create a new code every run because of the randomness
    identifier = sha256(
        f'{series_name}{episode_name}{random.random()}'.encode('utf-8')
    ).hexdigest()

    config_dict = {
        'id': identifier,
        'series_name': series_name,
        'episode_name': episode_name,
        'file': file,
        'source': {
            'url': f'https://www.youtube.com/watch?v={youtube_id}',
        },
    }
    return config_dict


def create_config_from_local(
    config_fname: str,
    base_path: Path = Path('data'),
    file_ext: str = '.gz',
    ignore_start: str | None = '_',
):
    my_json = dict(data=list())
    series_data = my_json['data']

    # Series are directories 
    if ignore_start is None: 
        series_glob_str = f'*/'
    else: # Ignore directories that start with certain substring
        series_glob_str = f'[!{ignore_start}]*/'
    all_series = base_path.glob(series_glob_str)
    for series_path in all_series:
        files = series_path.glob(f'*{file_ext}')
        series_name = str(series_path.stem)
        series_data.extend([
            single_config_file(fpath=file, series_name=series_name) 
            for file in files
        ])
    data_str = dumps(my_json, indent=2,)
    with open(config_fname, 'w') as file: file.write(data_str)

if __name__ == '__main__':
    OUTFILE = 'config.json'
    create_config_from_local(config_fname=OUTFILE)
