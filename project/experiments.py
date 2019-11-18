from databases.database import *
from models.des import *
import logging

logging.basicConfig(filename='des.log', filemode='w', format='%(name)s - %(levelname)s - %(message)s')


def run():
    series = [
        ('Gold', get_gold_serie()),
        ('Sunspot', get_suspot_serie()),
        ('DJI', get_DJI_serie())
    ]

    results = {}
    for name, serie in series:
        for auto in [True, False]:
            errors = []
            for i in range(30):
                ensemble = DES_PALR(get_gold_serie())
                error = ensemble.test(auto=auto, show_results=False, plot=False)
                errors.append(error)
            key = '%s %s' % (name, auto)
            results[key] = errors
    pd.DataFrame(results).to_csv('results.csv')

if __name__ == '__main__':
    run()

