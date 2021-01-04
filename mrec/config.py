CSV_FNAMES = {
    'validation': 'dataset/raw/validation.csv',
    'test': 'dataset/raw/test.csv',
}
DB_PATH = {
    'mrec': 'dataset/external/mrec.db'
}

class Constants:
    _unit_id = '_unit_id'
    relation = 'relation'
    direction = 'direction'
    sentence = 'sentence'
    term1 = 'term1'
    term2 = 'term2'

    @property
    def all_fields(self):
        return [getattr(self, k) for k, v in Constants().__dict__.items()]
