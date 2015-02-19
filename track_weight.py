#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import argparse
import logging
from trello import TrelloApi
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import vcr
import numpy as np
import numpy.polynomial.polynomial as poly
import os
from functools import partial

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

    errs = df.get('weight_errors', np.zeros_like(df['weight']))
    df.plot(y='weight', yerr=errs, ls='None', marker=marker, ax=axis, zorder=2,
            legend=False, color=colour)

    model = Model.from_dataframe(df)
    model.fit(degree=degree)
    if plot_model:
        model.plot(twin_ax)
    success_time = model.predict(target)
    if sigma_clip:
        try:
            sigma_clipped_out = df[~model.fitted_ind]
            sigma_clipped_out.plot(y='weight', ls='None', marker='o', ax=axis,
                                   color=fit_colour, zorder=3, legend=False)
        except TypeError:
            pass
    return success_time


def main(args):
    degree = 1
    with vcr.use_cassette('cassettes/query.yaml',
                          filter_headers=['authorization'],
                          filter_query_parameters=['token', 'key']):
        client = TrelloApi(apikey=TRELLO_API_KEY, token=TRELLO_APP_KEY)
        cards = client.lists.get_card(LIST_ID)

    weight, dt = [], []
    for card in cards:
        weight.append(float(card['name']))
        dt.append(create_date(card['id']))

    errors = np.ones_like(weight) * 0.05
    df = pd.DataFrame({'weight': weight, 'errors': errors}, index=dt)

    fig, axis = plt.subplots()
    newax = axis.twiny()

    success_time = analyse(df, axis, newax, degree=degree,
                           target=args.target, colour=colours[0],
                           fit_colour=colours[3], plot_model=False)

    resample = partial(df.resample, '1w',
            loffset=datetime.timedelta(days=-3.5))
    weekly = resample(how='mean')
    weekly_error = resample(how=lambda vals: vals.std() / np.sqrt(vals.size))
    weekly['weight_errors'] = weekly_error['weight']
    weekly.to_csv('weekly_binned.csv')

    success_time = analyse(weekly, axis, newax, degree=degree, marker='s',
                           target=args.target, colour=colours[1],
                           fit_colour=colours[4],
                           sigma_clip=False, plot_model=True)
    axis.set_title('Achieve target ({:.1f} kg) on {}'.format(
        args.target, success_time))

    newax.set_xticklabels([])

    sns.despine()
    fig.tight_layout()
    fig.savefig('weight_stats.pdf')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', help='Target weight', required=False,
                        type=float, default=63.5)
    main(parser.parse_args())
