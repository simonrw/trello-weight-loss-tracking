#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import argparse
import logging
import george
from george import kernels
import emcee
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import triangle
from multiprocessing import cpu_count

sns.set(style='white')

logging.basicConfig(level='DEBUG', format='%(asctime)s|%(name)s|%(levelname)s|%(message)s')
logger = logging.getLogger(__name__)

def model(params, t):
    _, _, m, c = params
    return m * t + c

def lnlike(p, t, y, yerr):
    a, tau = np.exp(p[:2])
    gp = george.GP(a * kernels.Matern32Kernel(tau))
    gp.compute(t, yerr)
    return gp.lnlikelihood(y - model(p, t))

def lnprior(p):
    lna, lntau, m, c = p
    if (-5 < lna < 5 and -5 < lntau < 5 and m < 0):
        return 0.
    return -np.inf

def lnprob(p, t, y, yerr):
    lp = lnprior(p)
    return lp + lnlike(p, t, y, yerr) if np.isfinite(lp) else -np.inf


def get_data():
    df = pd.read_csv('./weekly_binned.csv')
    x = (df.index - df.index.min()).values
    y = df.weight.values
    e = df.weight_errors.values

    return x, y, e

def main(args):
    logger.debug(args)
    x, y, e = get_data()
    ind = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    x, y, e = [data[ind] for data in [x, y, e]]

    fit = np.poly1d(np.polyfit(x, y, w=1. / e ** 2, deg=1))
    popt = fit.c
    initial = np.hstack([np.array([0., 0.]), popt])
    ndim = len(initial)
    nwalkers = args.nwalkers
    n_burn_in = args.nburn
    n_production = args.nprod
    p0 = [np.array(initial) + 1E-7 * np.random.randn(ndim)
            for i in xrange(nwalkers)]

    data = (x, y, e)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data,
            threads=cpu_count())
    logger.info('Running first burn in...')
    p1, _, _ = sampler.run_mcmc(p0, n_burn_in)
    sampler.reset()

    logger.info('Running production')
    sampler.run_mcmc(p1, n_production)

    samples = sampler.flatchain


    labels = [r'$\ln \alpha$', r'$\ln \tau$', r'$m$', r'$c$']
    fig = triangle.corner(samples, labels=labels)
    fig.savefig('gp_triangle.pdf')
    plt.close(fig)


    fig, axis = plt.subplots()
    axis.errorbar(x, y, e, ls='None', capsize=0., marker='o')

    sd_low, med, sd_high = np.percentile(samples[:, 2:], [15, 50, 84],
            axis=0)
    t = np.linspace(x.min(), x.max(), 50)
    med_model = model([None, None, med[0], med[1]], t)
    sd_low_model = model([None, None, sd_low[0], sd_low[1]], t)
    sd_high_model = model([None, None, sd_high[0], sd_high[1]], t)

    axis.fill_between(t, sd_low_model, sd_high_model, alpha=0.3)
    axis.plot(t, med_model, alpha=0.5)

    for s in samples[np.random.randint(len(samples), size=24)]:
        a, tau = np.exp(s[:2])
        gp = george.GP(a * kernels.Matern32Kernel(tau))
        gp.compute(x, e)

        m = gp.sample_conditional(y - model(s, x), t) + model(s, t)
        axis.plot(t, m, alpha=0.3, lw=0.5)

    axis.set_xlabel(r'Week')
    axis.set_ylabel(r'Weight / kg')
    fig.tight_layout()
    fig.savefig('gp_model.pdf')
    plt.close(fig)

    fig, axes = plt.subplots(ndim, 1, sharex=True)
    for (ax, chain, label) in zip(axes, samples.T, labels):
        ax.plot(chain, 'k-', lw=0.2)
        ax.set_ylabel(label)
    fig.tight_layout()
    axes[-1].set_xlabel(r'Step')
    fig.savefig('gp_chains.pdf')
    plt.close(fig)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--nburn',
            required=False, default=2000, type=int)
    parser.add_argument('-p', '--nprod',
            required=False, default=10000, type=int)
    parser.add_argument('-w', '--nwalkers',
            required=False, default=50, type=int)
    main(parser.parse_args())
