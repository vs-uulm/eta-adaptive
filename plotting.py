from matplotlib import pyplot as plt
from matplotlib import cm
from helpers import asqe, ecdf, e_survival
from statistics import mean
from scipy.stats import norm
from helpers import discretize_intervals, discretize_pointwise
import numpy as np


def plot_3d_fit(data, curves, titles):
    # preparation
    Ks = np.linspace(min(data["edges"]),max(data["edges"]))
    Ns = np.linspace(min(data["unique_nodes"]),max(data["unique_nodes"]))
    NsMG, KsMG = np.meshgrid(Ns, Ks)
    req_subplots = len(curves)
    fig = plt.figure(figsize=(15,15))
    for curve, title, i in zip(curves,titles, range(req_subplots)):
        ax = fig.add_subplot(200+(req_subplots>>1)*10+i+1, projection='3d')
        ax.set_title(title)
        ax.scatter(data["nodes"],data["edges"], data["mu_means"])
        ax.set_xlabel("Network Size")
        ax.set_ylabel("# Edges")
        ax.set_zlabel("Mu")
        surf = ax.plot_surface(NsMG, KsMG, curve(NsMG, KsMG), cmap=cm.coolwarm, antialiased=False)
    plt.show()


def discretization(freq, mu, si, nodes, edges):
    marksize = 100
    fig, axs = plt.subplots(1, 2, figsize=(15,15))
    plt.title(f"{nodes} nodes and {edges} edges")
    x_data = list(range(len(freq)))
    x_fits = np.linspace(0, len(freq)+5)
    pdfp = axs[0]
    cdfp = axs[1]
    dist = norm(mu, si)
    ppdf = dist.pdf(x_fits)
    disc_p = discretize_pointwise(dist)
    disc_nh = discretize_intervals(dist, halfstart=False)
    disc_h = discretize_intervals(dist)
    
    pdfp.bar(x_data, freq, label="Data", alpha=0.1)
    pdfp.plot(x_fits, ppdf, label="Predicted Normal")
    pdfp.scatter(range(len(disc_p)), disc_p, marker='x', s=marksize, label="Pointwise")
    #pdfp.scatter(range(len(disc_nh)), disc_nh, marker='+', s=marksize, label="Interval 0-1")
    pdfp.scatter(range(len(disc_h)), disc_h, marker='x', s=marksize, label="Interval 0.5-1.5")
    #pdfp.legend()
    
    pcdf = dist.cdf(x_fits)
    freqsum = [sum(freq[:min(_,len(freq))]) for _ in range(len(freq)+5)]
    cdfp.bar(range(len(freq)+5), freqsum, label="Data", alpha=0.1)
    cdfp.plot(x_fits, pcdf, label="Predicted Normal")
    cdfp.scatter(range(len(disc_p)), [sum(disc_p[:_]) for _ in range(len(disc_p))], marker='x', s=marksize, label="Pointwise")
    #cdfp.scatter(range(len(disc_nh)), [sum(disc_nh[:_]) for _ in range(len(disc_nh))], marker='+', s=marksize, label="Interval 0-1")
    cdfp.scatter(range(len(disc_h)), [sum(disc_h[:_]) for _ in range(len(disc_h))], marker='x', s=marksize, label="Interval 0.5-1.5")
    cdfp.legend(bbox_to_anchor=(1.04,1), loc="upper left")


def fit_errors(titles, datas, models, xlabel, ylabel, logscale=True):
    ln = len(datas)
    fig, axs = plt.subplots(1, ln, sharey=True, figsize=(15,15))
    
    errors = [[] for _ in range(ln)]
    for data, errs in zip(datas,errors):
        for model in models:
            errs.append(asqe(data, model))
    
    for i in range(ln):
        axs[i].boxplot(errors[i])
        if logscale:
            axs[i].set_yscale("log")
        axs[i].set_xlabel(xlabel)
        axs[i].set_ylabel(ylabel)
        axs[i].set_title(titles[i])
    return fig


def plot_and_compare_cdf(data, rv, suptitle=None, plot_confidence=False):
    fig = plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.plot(data, ecdf(data), marker='.', markersize=4, color='r')
    plt.plot(data, rv.cdf(data), color='b')
    if plot_confidence:
        df = get_cdf_confidence(data)
        nrows, ncols = df.shape
        low = int(ncols * 0.025)
        high = int(ncols * 0.975)
        plt.fill_between(df.index, df[low], df[high], alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(2*10**(-4),1.05)
    plt.grid(True)

    plt.subplot(1,3,2)
    plt.plot(data, ecdf(data), color='r')
    plt.plot(data, rv.cdf(data), color='b')
    plt.ylim(-0.01,1.01)
    plt.grid(True)

    plt.subplot(1,3,3)
    plt.plot(data, e_survival(data), marker='.', markersize=4, color='r')
    plt.plot(data, rv.sf(data), color='b')
    if plot_confidence:
        df = get_surv_confidence(data)
        nrows, ncols = df.shape
        low = int(ncols * 0.025)
        high = int(ncols * 0.975)
        plt.fill_between(df.index, df[low], df[high], alpha=0.2)
    plt.xscale('log')
    plt.yscale('log')
    plt.ylim(2*10**(-4),1.05)
    plt.grid(True)
    if suptitle is not None:
        fig.suptitle(suptitle, fontsize=20)
        