import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)


def get_gold_serie(diff=False):
    url='https://raw.githubusercontent.com/FinYang/tsdl/master/data-raw/commod/gold.dat'
    s = requests.get(url).content
    data = ' '.join([l.strip() for l in s.decode('utf-8').split('\n')[1:]])
    serie = pd.Series(
        data.split(' '),
        index=pd.date_range(start ='1-1-1900', end ='4-7-1997', freq ='Y')
    )
    serie = serie.astype(float)
    if diff:
        return np.diff(serie)
    return serie


def get_suspot_serie(diff=False):
    url='https://github.com/jamissonfreitas/IN1167/raw/master/databases/sunspot.txt'
    s = requests.get(url).content
    data = ' '.join([l.strip() for l in s.decode('utf-8').split('\n')])
    data = data.split(' ')[:-1]
    serie = pd.Series(
        data,
        index=pd.date_range(start ='1-1-2000', end ='11-09-2000', freq ='D')
    )
    serie = serie.dropna()
    serie = serie.astype(float)
    if diff:
        return np.diff(serie)
    return serie


def get_DJI_serie(diff=False):
    url='https://raw.githubusercontent.com/jamissonfreitas/IN1167/master/databases/DJI.csv'
    df = pd.read_csv(url)
    indexs = pd.to_datetime(df.Date, format='%Y-%m-%d')
    df.index = indexs
    if diff:
        return np.diff(df.Close)
    return df.Close


def get_airline(diff=False):
    url='https://raw.githubusercontent.com/jamissonfreitas/IN1167/master/database/airline.csv'
    df = pd.read_csv(url)
    serie = df.Passengers
    serie = serie.astype(float)
    if diff:
        return np.diff(serie)
    return serie


def get_star(diff=False):
    pass


def get_STLFSI(diff=False):
    url='https://raw.githubusercontent.com/jamissonfreitas/IN1167/master/databases/STLFSI.csv'
    df = pd.read_csv(url)
    serie = df.STLFSI
    serie = serie.astype(float)
    if diff:
        return np.diff(serie)
    return serie
