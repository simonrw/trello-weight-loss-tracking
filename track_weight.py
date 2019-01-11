#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import argparse
import logging
from trello import TrelloApi
import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import seaborn as sns
import vcr
import numpy as np
import numpy.polynomial.polynomial as poly
import os
from functools import partial

EVENTS = [
    {'start': datetime.datetime(2016, 2, 17),
     'end': datetime.datetime(2016, 2, 19),
     'label': 'Geneva meeting',
    },
    {'start': datetime.datetime(2016, 2, 19),
     'end': datetime.datetime(2016, 3, 4),
     'label': 'Chile trip',
    },
    {'start': datetime.datetime(2016, 8, 13),
     'end': datetime.datetime(2016, 8, 20),
     'label': 'Wedding',
    }
]

sns.set(style='white')


logging.basicConfig(
    level=logging.WARNING, format='%(asctime)s|%(name)s|%(levelname)s|%(message)s')
logger = logging.getLogger(__name__)

TRELLO_APP_KEY = os.environ['TRELLO_APP_KEY']
TRELLO_API_KEY = os.environ['TRELLO_API_KEY']
BOARD_ID = os.environ['BOARD_ID']
LIST_ID = os.environ['LIST_ID']

colours = sns.color_palette('cubehelix', n_colors=7)


def create_date(object_id):
    return datetime.datetime.fromtimestamp(int(object_id[:8], 16))


class Model(object):

    def __init__(self, x, y, e):
        self.x = x
        self.y = y
        self.e = e
        self.model = None
        self.t = np.linspace(self.x.min(), self.x.max(), 500)

    def fit(self, degree):
        self.model = poly.Polynomial(poly.polyfit(self.x, self.y, degree,
                                                  w=1. / self.e ** 2))
        residuals = self.y - self.model(self.x)
        self.fitted_ind = np.abs(residuals) < 2 * residuals.mad()
        self.model = poly.Polynomial(poly.polyfit(
            self.x[self.fitted_ind], self.y[self.fitted_ind], degree,
            w=1. / self.e[self.fitted_ind] ** 2))

        return self

    def predict(self, target_weight):
        fit_coeffs = self.model.coef.copy()
        fit_coeffs[0] -= target_weight
        roots = poly.Polynomial(fit_coeffs).roots()
        real_roots = [root for root in roots if root.imag == 0]
        if len(real_roots) == 1:
            ndays = real_roots[0].real
            return (self.df.index.min() +
                    datetime.timedelta(days=ndays)).date()
        else:
            return '<Unknown>'

    def plot(self, axis):
        fit_coeffs = self.model.coef.copy()
        yvals = self.model(self.t)
        axis.plot(self.t, yvals, color=colours[1])
        return self

    @classmethod
    def from_dataframe(cls, df):
        x = df.index.to_julian_date()
        x -= x.min()
        y = df.weight
        e = df.errors
        self = cls(x, y, e)
        self.df = df
        return self


def analyse(df, axis, twin_ax, degree, target, colour, fit_colour,
            sigma_clip=True, marker='o', plot_model=True):

    errs = df['errors']
    df.plot(y='weight', yerr=errs, ls='None', marker=marker, ax=axis, zorder=2,
            legend=False, color=colour, alpha=0.2)

    # model = Model.from_dataframe(df)
    # model.fit(degree=degree)
    # if plot_model:
    #     model.plot(twin_ax)
    # success_time = model.predict(target)
    # if sigma_clip:
    #     try:
    #         sigma_clipped_out = df[~model.fitted_ind]
    #         sigma_clipped_out.plot(y='weight', ls='None', marker='o', ax=axis,
    #                                color=fit_colour, zorder=3, legend=False)
    #     except TypeError:
    #         pass
    # return success_time

class DateRectangle(plt.Rectangle):
    @classmethod
    def from_dates(cls, start_date, end_date, low, height, *args, **kwargs):
        start = mdates.date2num(start_date)
        end = mdates.date2num(end_date)
        width = end - start
        return cls((start, low), width, height, *args, **kwargs)

def add_events(axis):
    print(axis)
    low = 94
    height = 0.5
    for event in EVENTS:
        rect = DateRectangle.from_dates(
            start_date=event['start'],
            end_date=event['end'],
            low=low, height=height,
            facecolor='yellow', edgecolor='black', linewidth=1)

        axis.add_patch(rect)

        rx, ry = rect.get_xy()
        cx = rx + rect.get_width() / 2.
        cy = ry + rect.get_height() / 2.

        axis.annotate(event['label'], (cx, cy), color='k',
                      fontsize=8, ha='center', va='center')



def main(args):
    degree = 1
    cassette_name = 'cassettes/query.yaml'
    if args.reset and os.path.isfile(cassette_name):
        os.remove(cassette_name)

    with vcr.use_cassette('cassettes/query.yaml',
                          filter_headers=['authorization'],
                          filter_query_parameters=['token', 'key']):
        client = TrelloApi(apikey=TRELLO_API_KEY, token=TRELLO_APP_KEY)
        cards = client.lists.get_card(LIST_ID)

    weight, dt, errors = [], [], []
    for card in cards:
        try:
            weight.append(float(card['name']))
        except ValueError:
            measurements = [float(value) for value in card['name'].split()]
            mean_value = np.average(measurements)
            error_value = np.std(measurements) / np.sqrt(len(measurements))
            weight.append(mean_value)
            errors.append(error_value)
        else:
            errors.append(0.)

        dt.append(create_date(card['id']))

    df = pd.DataFrame({'weight': weight, 'errors': errors}, index=dt)

    # Add a base level of measurement error
    measurement_error = 0.05
    df['errors'] = np.sqrt(df['errors'] ** 2 + measurement_error ** 2)

    fig, axis = plt.subplots()
    newax = axis.twiny()

    success_time = analyse(df, axis, newax, degree=degree,
                           target=args.target, colour=colours[0],
                           fit_colour=colours[3], plot_model=False)

    resample = partial(df.resample, '1m',
            loffset=datetime.timedelta(days=-3.5))
    weekly = resample(how='mean')
    weekly_error = resample(how=lambda vals: vals.std() / np.sqrt(vals.size))
    weekly['weight_errors'] = weekly_error['weight']
    # Include measurement error
    weekly['weight_errors'] = np.sqrt(weekly['weight_errors'] ** 2 + 0.01)
    weekly.to_csv('weekly_binned.csv')
    axis.errorbar(weekly.index, weekly.weight, weekly.weight_errors,
            ls='None', marker='s')

    # success_time = analyse(weekly, axis, newax, degree=degree, marker='s',
    #                        target=args.target, colour=colours[1],
    #                        fit_colour=colours[4],
    #                        sigma_clip=False, plot_model=True)
    # axis.set_title('Achieve target ({:.1f} kg) on {}'.format(
    #     args.target, success_time))

    newax.set_xticklabels([])

    add_events(axis)

    sns.despine()
    fig.tight_layout()
    fig.savefig('weight_stats.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', help='Target weight', required=False,
                        type=float, default=63.5)
    parser.add_argument('-r', '--reset', action='store_true',
            default=False, required=False, help='Reset cassettes')
    main(parser.parse_args())
