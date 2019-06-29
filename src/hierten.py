#!/usr/bin/python3.6
# -*- coding=utf-8 -*-

#  Project: catchcore
#    Class: Hierten
#         Hierarchical dense sub-tensor detection class
#
#    hierten.py
#      Version:  1.0
#      Goal: Class script
#      Created by @wenchieh  on <11/18/2018>
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <11/18/2018>
#


__author__ = 'wenchieh'

# sys
from copy import *
import collections

# third-party lib
import numpy as np
import numpy.linalg as nplg


class HierTen(object):
    # input para.
    T = None  # the input tensor
    ndim = None  # the dimension of input tensor
    shape = None  # input tensor shape
    # alg. para.
    ps = 0.0  # penalty para. for missing entities p > 0
    hs = -1  # the number of hierarchies (layers)
    beta = 0  # constraints para. (regularization term coef.)
    etas = None  # the density constraints between layers (gradually increasing)
    # res para.
    valid_hs = -1
    hsindicator = None
    hsdensity = None
    hnnzs = None

    def __init__(self, tensor):
        if len(tensor.vals) <= 0:
            ValueError("the input tensor is ZERO!")

        self.T = tensor
        self.shape = tensor.shape
        self.ndim = tensor.ndim

    def setting(self, hierarchies=None, penalties=None, constraintpara=None, etas=None):
        self.hs = hierarchies if hierarchies is not None else 10
        self.beta = constraintpara if constraintpara is not None else 1.8
        self.valid_hs = self.hs

        if penalties is None:
            self.ps = [1e-3] * self.hs
        elif not isinstance(penalties, collections.Iterable):
            self.ps = [penalties] * self.hs
        elif isinstance(penalties, collections.Iterable):
            if len(penalties) == self.hs:
                self.ps = penalties
            elif len(penalties) == 1:
                self.ps = [penalties[0]] * self.hs

        if etas is None:
            halfhs = int(np.round(self.hs / 2.0))
            self.etas = np.asarray([1.5] * halfhs + [1.1] * (self.hs - halfhs))
        elif not isinstance(etas, collections.Iterable):
            self.etas = np.asarray([etas] * self.hs)
        elif isinstance(etas, collections.Iterable):
            if len(etas) == self.hs:
                self.etas = np.asarray(etas, float)
            elif len(etas) == 1:
                self.etas = np.asarray([etas[0]] * self.hs)
        else:
            ValueError(
                "input parameter etas is invalid, it must be a float or a list with size as hierarchies!")
        return

    def _density_(self, xs):
        density = self.T.ttv(tuple(xs), modes=range(self.ndim))
        density /= np.prod([np.sum(x) * 1.0 for x in xs])
        return density

    def _subtensor_nnzs_(self, indicators):
        xs, shape = list(), list()
        for dm in range(self.ndim):
            if len(indicators[dm]) > 0:
                xdm = np.zeros((self.shape[dm]))
                xdm[indicators[dm]] = 1
            else:
                xdm = np.ones((self.shape[dm]))
            xs.append(xdm)
            shape.append(np.sum(xdm, dtype=int))
        nnzs = self.T.ttv(tuple(xs), modes=range(self.ndim))

        return int(nnzs), tuple(shape)

    def _OBJhierdense_(self, xs, h, Ch):
        '''
        the objective function for measuring the density of sub-tensor induced by indicators [xs]
        :param xs: indicators for each dimension
        :param h: h-th hierarchy of dense tensor, h >= 0
        :param Ch: the density of h-th hierarchy
        :return: the density
        '''
        objval = 0.0
        ttvxs = self.T.ttv(tuple(xs), modes=range(self.ndim))
        objval += -1.0 * (1 + self.ps[h] + (h > 0) * self.beta) * ttvxs
        objval += 1.0 * (self.ps[h] + (h > 0) * Ch * self.beta) * \
            np.prod([np.sum(x) * 1.0 for x in xs])
        return objval

    def _GRADhierdense_(self, xs, dim, h, Ch):
        '''
        the gradient for the objective function  of hierarchical density
        :param xs: indicators for each dimension
        :param dim: the dimension for focusing
        :param h: h-th hierarchy of dense tensor, h >= 0
        :param Ch: the density of h-th hierarchy
        :return:
        '''
        xs_nh = xs[:dim] + xs[dim + 1:]
        scale = np.prod([np.sum(x) * 1.0 for x in xs_nh])
        grad_xdim = np.array([0.0] * self.shape[dim])
        ttvxs = self.T.ttv(tuple(xs_nh), modes=range(dim) + range(dim + 1, self.ndim))
        grad_xdim += -1.0 * (1 + self.ps[h] + (h > 0) * self.beta) * np.squeeze(ttvxs.toarray())
        grad_xdim += 1.0 * (self.ps[h] + (h > 0) * Ch * self.beta) * \
            scale * np.ones((self.shape[dim]))
        return grad_xdim

    def _projected_grads_(self, x, gradx, lb, ub):
        # project the gradient for x to lower and upper bounds.
        grad = copy(gradx)
        lbxs = np.where(x == lb)[0]
        grad[lbxs] = np.minimum(0, grad[lbxs])
        ubxs = np.where(x == ub)[0]
        grad[ubxs] = np.maximum(0, grad[ubxs])
        return grad

    def _projected_optimize_(self, xs_old, lb, ub, dim, h, Ch, maxiter=5, tol=1e-6):
        """
        projected gradient descent optimization algorithm
        to solve the each optimization sub-problem: the h-th hierarchy h \in [0, self.hs-1]
        :param xs_old: the indicators vector from last update.
        :param lb: the lower bound for the indicator vector
        :param ub: the upper bound for the indicator vector
        :param dim: the optimization dimension
        :param h: the h-th hierarchy
        :param Ch: the density value for the last hierarchy
        :param maxiter: the maximum number of updating iteration
        :param tol: the tolerance value for updating (stop criteria)
        :return: the update gradient vector for x
        """

        alpha = 1  # the initial step size
        sigma = 0.01  # para. for the line search to select a good step size
        ratio = 0.1  # ratio of the chane of the step size
        ubalpha = 10
        succ = 0

        # projected gradient descent alg.
        iter = 0
        old = copy(xs_old)
        new = None
        itercond = True
        while (iter < maxiter):  # to be convergence
            if iter > 0:
                old = copy(new)

            objold = self._OBJhierdense_(old, h, Ch)
            grad = self._GRADhierdense_(old, dim, h, Ch)

            # searching for step-size that satisfies the modified Armijo condition
            while succ <= ubalpha and itercond:
                xsdm_new = np.minimum(np.maximum(old[dim] - alpha * grad, lb),
                                      ub)  # projected to bounded feasible region
                # [usually not entered]
                # spacial case processing for xsdm_new = 0, to make sure that u_new = u - eta*grad_u >= 0
                if np.sum(np.abs(xsdm_new)) == 0:
                    xsdm_gradpos = grad > 0
                    alpha = np.mean(old[dim][xsdm_gradpos] / grad[xsdm_gradpos])
                    xsdm_new = np.minimum(np.maximum(old[dim] - alpha * grad, lb), ub)

                # Armijo alg.
                succ += 1
                new = old[:dim] + [xsdm_new] + old[dim + 1:]
                objnew = self._OBJhierdense_(new, h, Ch)
                # Armijo's line search condition
                itercond = (objnew - objold > sigma * np.dot(grad, (new[dim] - old[dim])))
                if itercond:
                    alpha *= ratio
                else:
                    alpha /= np.sqrt(ratio)

            iter += 1
            if nplg.norm(new[dim] - old[dim]) < tol:  # for convergence update
                break
        # the updated indicator vector for the given dimension
        return new[dim]

    def _optimal_hiertens(self, selects=None, dimension=None, maxiter=100, eps=1e-7, convtol=1e-6, debug=False):
        '''
        get the optimal hierarchical dense sub-tensors
        :param selects: [array | list, default=None], the selected seed for specific focus
        :param maxiter: [int, default=100], the maxiimum iterations
        :param eps: [float, default=1e-7], the tolerance threshold
        :param convtol: [float, default=1e-5], the threshold for convergence.
        :return:
        '''
        if selects is None or dimension is None:
            selects, dimension = None, None

        # initialization for Xs and grad-norm
        xhs = list()
        for h in range(self.hs):
            xhs.append([0.01 * np.ones((dm)) for dm in self.shape])
        for dm in range(self.ndim):
            xhs[0][dm] = 0.5 * np.ones((self.shape[dm]))

        # the average density for the whole tensor by selecting each element
        C0 = self.etas[0] * np.sum(self.T.vals) * 1.0 / np.prod(np.asarray(self.shape, float))
        # compute the initial gradients for each indicator vector
        grad_x0, grad_xk = list(), list()
        for dm in range(self.ndim):
            grad_x0.append(self._GRADhierdense_(xhs[0], dm, 0, 0))
            grad_xk.append(self._GRADhierdense_(xhs[1], dm, 1, C0))

        # initial projected gradient norm
        # lower and upper bounds for indicator vectors
        initnorm = 0.0
        lbs = xhs[1]
        if selects is not None:
            lbs[dimension][np.array(selects, dtype=int)] = 0.9
        ubs = [np.ones((dm)) for dm in self.shape]
        xhs_gradproj = list()
        for dm in range(self.ndim):
            init_xsp = np.zeros((self.shape[dm], self.hs))
            init_xsp[:, 0] = self._projected_grads_(xhs[0][dm], grad_x0[dm], lbs[dm], ubs[dm])
            xskp = self._projected_grads_(xhs[1][dm], grad_xk[dm], lbs[dm], ubs[dm])
            init_xsp[:, 1:] = np.repeat(np.asarray([xskp]), self.hs - 1, 0).T
            xhs_gradproj.append(init_xsp)
            initnorm += nplg.norm(init_xsp[:, 0]) ** 2 + (self.hs -
                                                          1) * nplg.norm(init_xsp[:, 1]) ** 2
        initnorm = np.sqrt(initnorm)

        # iterational optimization
        # dimensional alternative projected gradient descent optimization
        iter = 0
        norm_trace = list([initnorm])
        while (iter < maxiter):
            if iter % 10 == 0:
                print("iteration: %d" % iter)
            xh0_old = deepcopy(xhs[0])
            for dm in range(self.ndim):
                lbx, ubx = xhs[1][dm], np.ones((self.shape[dm],))
                xhs[0][dm] = self._projected_optimize_(xh0_old, lbx, ubx, dm, 0, 0, tol=convtol)
                grad_xsdm = self._GRADhierdense_(xhs[0], dm, 0, 0)
                xhs_gradproj[dm][:, 0] = self._projected_grads_(xhs[0][dm], grad_xsdm, lbx, ubx)
                xh0_old = deepcopy(xhs[0])

            # update for the each hierarchy. i.e. k \in [1, hs)
            for h in range(1, self.hs):
                C = self.etas[h] * self._density_(xhs[h - 1])
                xhs_old = deepcopy(xhs[h])
                # solve the subproblem of xhs[h], and update the dimension alternatively.
                for dm in range(self.ndim):
                    # lower bound as the current next hierarchy indicator vector, i.e. nested node constraint.
                    lbs_xdm = xhs[h + 1][dm] if h < self.hs - 1 else np.zeros((self.shape[dm],))
                    if selects is not None and dimension == dm:
                        lbs_xdm[np.asarray(selects, int)] = 0.9
                    # lower bound as the last hierarchy indicator vector.
                    ubs_xdm = xhs[h - 1][dm]
                    xhs[h][dm] = self._projected_optimize_(
                        xhs_old, lbs_xdm, ubs_xdm, dm, h, C, tol=convtol)
                    grad_xsdm_new = self._GRADhierdense_(xhs[h], dm, h, C)
                    xhs_gradproj[dm][:, h] = self._projected_grads_(
                        xhs[h][dm], grad_xsdm_new, lbs_xdm, ubs_xdm)

            iter += 1
            # early stopping (criteria)
            norm_new = 0.0
            for dm in range(self.ndim):
                norm_new += nplg.norm(xhs_gradproj[dm]) ** 2
            norm_new = np.sqrt(norm_new)
            norm_trace.append(norm_new)
            if norm_new < eps * initnorm:
                break

        if iter >= maxiter:
            print("Warning: maximum iterators in nls subproblem")
        if debug:
            print("# iters: {}, norm_init: {}, norm_final: {}".format(
                iter, initnorm, norm_trace[-1]))
            print(norm_trace)
        return xhs

    def hieroptimal(self, maxiter=100, eps=1e-7, convtol=1e-5, debug=False):
        '''
        hierarchical optimization for dense sub-tensor detection
        :param maxiter: the maximum number of updating iteration
        :param eps: the tolerance value for updating (stop criteria)
        :param convtol: the threshold for convergence.
        :return:
        '''
        return self._optimal_hiertens(maxiter=maxiter, eps=eps, convtol=convtol, debug=debug)

    def queryhiers(self, seeds, dimension, maxiter=100, eps=1e-7, convtol=1e-5):
        '''
        query specific hierarchical dense sub-densors
        :param seeds: queried seeds
        :param dimension: selected dimension
        :return:
        '''
        if seeds is None or dimension is None:
            seeds, dimension = None, None
            ValueError("Neither of the dimension and seeds can be [None].")
        if dimension not in range(self.ndim):
            ValueError("selected dimension must be in [0, %d)." % (self.ndim))
        return self._optimal_hiertens(seeds, dimension, maxiter, eps, convtol)

    def hierarchy_indicator(self, optxs, tholds=0.5):
        nhier = len(optxs)
        thetas = list()
        validhs = list()
        indicators = list()
        hnnzs, hshapes, hdensities = list(), list(), list()

        if isinstance(tholds, collections.Iterable):
            if len(tholds) < self.ndim:
                thetas = [tholds[0]] * self.ndim
            else:
                thetas = tholds
        elif tholds is None or not isinstance(tholds, collections.Iterable):
            thetas = [tholds] * self.ndim

        for h in range(nhier):
            print("H{}:".format(h + 1))
            print("{}".format([(np.min(optxs[h][dm]), np.max(optxs[h][dm]))
                               for dm in range(self.ndim)]))
            hidxs, hshp = list(), list()
            isdiff = 0
            for dm in range(self.ndim):
                dmidx = np.where(optxs[h][dm] > thetas[dm])[0]
                hidxs.append(sorted(dmidx))
                hshp.append(len(dmidx))
                if h > 0 and len(indicators) > 0:
                    isdiff += int(set(indicators[-1][dm]) != set(dmidx))
            cards = np.prod(np.asarray(hshp, float))
            if cards > 0 and (h == 0 or (h > 0 and isdiff > 0)):
                nnz, _ = self._subtensor_nnzs_(hidxs)
                if h == 0 or (h > 0 and nnz != hnnzs[-1]):
                    validhs.append(h)
                    indicators.append(hidxs)
                    hnnzs.append(nnz)
                    hshapes.append(tuple(hshp))
                    hdensities.append(nnz * 1.0 / cards)

        self.valid_hs = len(validhs)
        self.hsindicator = indicators

        return validhs, indicators, hnnzs, hshapes, hdensities

    def dump(self):
        print("Basic Information:")
        print("  n_dims: {}, shape: {}, nnzs: {}, totals: {}".format(self.ndim, self.shape,
                                                                     len(self.T.vals), np.sum(self.T.vals)))
        hiershape_str = ""
        for hs in range(self.valid_hs):
            hshp_str = "(" + ",".join(map(str, [len(self.hsindicator[hs][dm])
                                                for dm in range(self.ndim)])) + ")"
            hiershape_str += hshp_str + ',  '
        print("  Valid hierarchies:{}, shapes:{}".format(self.valid_hs, hiershape_str[:-3]))

        if self.hsdensity is not None:
            print(
                "Hierarchical densities and none-zeros (average density: {}):".format(self.hsdensity[0]))
            print(" \t{}".format(self.hsdensity[1:]))
            print(" \t{}".format(self.hnnzs))
        print("done!")
