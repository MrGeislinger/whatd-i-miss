from pathlib import Path
from json import dumps

def single_config_file(fpath: Path, series_name: str) -> dict:
    file = str(fpath.relative_to('.'))
    episode_name = str(fpath.stem)
    youtube_id = episode_name.split('.')[-1]

    config_dict = {
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
):
    my_json = dict(data=list())
    series_data = my_json['data']
    # Series are directory
    all_series = base_path.glob('*/')
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
