#!/usr/bin/python
# -*- coding=utf-8 -*-

#  Project: catchcore
#    Class: Hierten
#         Hierarchical dense sub-tensor detection class
#
#    hierten.py
#      Version:  2.0
#      Goal: Class script
#      Created by @wenchieh  on <12/10/2019>
#
#      Copyright:
#        This software is free of charge under research purposes.
#        For commercial purposes, please contact the author.
#
#      Created by @wenchieh  on <12/10/2019>
#

__author__ = 'wenchieh'


# sys
from copy import copy, deepcopy
import collections

# third-party lib
import numpy as np
import numpy.linalg as nplg
from sktensor import cp_als


class HierTen(object):
    # input para.
    T = None     # the input tensor
    n_dim = None  # the dimension of input tensor
    shape = None  # input tensor shape
    # alg. para.
    ps = 0.0      # penalty para. for missing entities p > 0
    hs = -1      # the number of hierarchies (layers)
    lamb = 0     # constraints para. (regularization term coeff.)
    etas = None  # the density constraints between layers (gradually increasing)
    # res para.
    valid_hs = -1
    hs_indicator = None
    hs_density = None
    hs_nnz = None

    def __init__(self, tensor):
        if len(tensor.vals) <= 0:
            ValueError("the input tensor is ZERO!")

        self.T = tensor
        self.shape = tensor.shape
        self.n_dim = tensor.ndim

    def setting(self, hierarchies=None, penalties=None, constraint_para=None, etas=None):
        self.hs = hierarchies if hierarchies is not None else 10
        self.lamb = constraint_para if constraint_para is not None else 1.8
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
            half_hs = int(np.round(self.hs / 2.0))
            self.etas = np.array([1.5] * half_hs + [1.1] * (self.hs - half_hs))
        elif not isinstance(etas, collections.Iterable):
            self.etas = np.array([etas] * self.hs)
        elif isinstance(etas, collections.Iterable):
            if len(etas) == self.hs:
                self.etas = np.array(etas, float)
            elif len(etas) == 1:
                self.etas = np.array([etas[0]] * self.hs)
        else:
            ValueError("input parameter etas is invalid, it must be a float or a list with size as hierarchies!")
        return

    def _density_(self, xs):
        density = self.T.ttv(tuple(xs), modes=range(self.n_dim))
        density /= np.prod([np.sum(x) * 1.0 for x in xs])
        return density

    def _subtensor_nnzs_(self, indicators):
        xs, shape = list(), list()
        for dm in range(self.n_dim):
            if len(indicators[dm]) > 0:
                xdm = np.zeros(self.shape[dm])
                xdm[indicators[dm]] = 1
            else:
                xdm = np.ones(self.shape[dm])
            xs.append(xdm)
            shape.append(np.sum(xdm, dtype=int))
        nnzs = self.T.ttv(tuple(xs), modes=range(self.n_dim))

        return int(nnzs), tuple(shape)

    def _OBJhierdense_(self, xs, h, Ch):
        '''
        the objective function for measuring the density of sub-tensor induced by indicators [xs]
        :param xs: indicators for each dimension
        :param h: h-th hierarchy of dense tensor, h >= 0
        :param Ch: the density of h-th hierarchy
        :return: the density
        '''
        obj_val = 0.0
        ttvxs = self.T.ttv(tuple(xs), modes=range(self.n_dim))
        obj_val += -1.0 * (1 + self.ps[h] + (h > 0) * self.lamb) * ttvxs
        obj_val += 1.0 * (self.ps[h] + (h > 0) * Ch * self.lamb) * np.prod([np.sum(x) * 1.0 for x in xs])
        return obj_val

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
        ttv_xs = self.T.ttv(tuple(xs_nh), modes=list(range(dim)) + list(range(dim + 1, self.n_dim)))
        grad_xdim += -1.0 * (1 + self.ps[h] + (h > 0) * self.lamb) * np.squeeze(ttv_xs.toarray())
        grad_xdim += 1.0 * (self.ps[h] + (h > 0) * Ch * self.lamb) * scale * np.ones((self.shape[dim]))
        return grad_xdim

    def _projected_grads_(self, x, grad_x, lb, ub):
        # project the gradient for x to lower and upper bounds.
        grad = copy(grad_x)
        lb_xs = np.where(x == lb)[0]
        grad[lb_xs] = np.minimum(0, grad[lb_xs])
        ub_xs = np.where(x == ub)[0]
        grad[ub_xs] = np.maximum(0, grad[ub_xs])
        return grad

    def _projected_optimize_(self, xs_old, lb, ub, dim, h, Ch, max_iters=5, tol=1e-6):
        """
        projected gradient descent optimization algorithm
        to solve the each optimization sub-problem: the h-th hierarchy h \in [0, self.hs-1]
        :param xs_old: the indicators vector from last update.
        :param lb: the lower bound for the indicator vector
        :param ub: the upper bound for the indicator vector
        :param dim: the optimization dimension
        :param h: the h-th hierarchy
        :param Ch: the density value for the last hierarchy
        :param max_iters: the maximum number of updating iteration
        :param tol: the tolerance value for updating (stop criteria)
        :return: the update gradient vector for x
        """

        alpha = 1      # the initial step size
        sigma = 0.01   # para. for the line search to select a good step size
        ratio = 0.1    # ratio of the change for the step size
        ub_alpha = 10
        n_succ = 0

        # projected gradient descent alg.
        n_iter = 0
        old = copy(xs_old)
        new = None
        iter_cond = True
        while(n_iter < max_iters):   # to be convergence
            if n_iter > 0:
                old = copy(new)

            obj_old = self._OBJhierdense_(old, h, Ch)
            grad = self._GRADhierdense_(old, dim, h, Ch)

            # searching for step-size that satisfies the modified Armijo condition
            while n_succ <= ub_alpha and iter_cond:
                xsdm_new = np.minimum(np.maximum(old[dim] - alpha * grad, lb), ub)  # projected to bounded feasible region
                # [usually not entered]
                # spacial case processing for xsdm_new = 0, to make sure that u_new = u - eta*grad_u >= 0
                if np.sum(np.abs(xsdm_new)) == 0:
                    xsdm_gradpos = grad > 0
                    alpha = np.mean(old[dim][xsdm_gradpos] / grad[xsdm_gradpos])
                    xsdm_new = np.minimum(np.maximum(old[dim] - alpha * grad, lb), ub)

                # Armijo alg.
                n_succ += 1
                new = old[:dim] + [xsdm_new] + old[dim + 1:]
                obj_new = self._OBJhierdense_(new, h, Ch)
                # Armijo's line search condition
                iter_cond = (obj_new - obj_old > sigma * np.dot(grad, (new[dim] - old[dim])))
                if iter_cond:
                    alpha *= ratio
                else:
                    alpha /= np.sqrt(ratio)

            n_iter += 1
            if nplg.norm(new[dim] - old[dim]) < tol:   # for convergence update
                break
        # the updated indicator vector for the given dimension
        return new[dim]

    def _optimal_hiertens(self, max_iters=100, eps=1e-7, convtol=1e-6, debug=False):
        '''
        get the optimal hierarchical dense sub-tensors
        :param max_iters: [int, default=100], the maximum iterations
        :param eps: [float, default=1e-7], the tolerance threshold
        :param convtol: [float, default=1e-5], the threshold for convergence.
        :return:
        '''
        xhs = list()
        for h in range(self.hs):
            xhs.append([0.01 * np.ones((dm)) for dm in self.shape])
        for dm in range(self.n_dim):
            xhs[0][dm] = 0.5 * np.ones((self.shape[dm]))

        # the average density for the whole tensor by selecting each element
        C0 = self.etas[0] * np.sum(self.T.vals) * 1.0 / np.prod(np.array(self.shape, float))
        # compute the initial gradients for each indicator vector
        grad_x0, grad_xk = list(), list()
        for dm in range(self.n_dim):
            grad_x0.append(self._GRADhierdense_(xhs[0], dm, 0, 0))
            grad_xk.append(self._GRADhierdense_(xhs[1], dm, 1, C0))

        # initial projected gradient norm
        # lower and upper bounds for indicator vectors
        init_norm = 0.0
        lbs = xhs[1]
        ubs = [np.ones((dm)) for dm in self.shape]
        xhs_grad_proj = list()
        for dm in range(self.n_dim):
            init_xsp = np.zeros((self.shape[dm], self.hs))
            init_xsp[:, 0] = self._projected_grads_(xhs[0][dm], grad_x0[dm], lbs[dm], ubs[dm])
            xskp = self._projected_grads_(xhs[1][dm], grad_xk[dm], lbs[dm], ubs[dm])
            init_xsp[:, 1:] = np.repeat(np.array([xskp]), self.hs - 1, 0).T
            xhs_grad_proj.append(init_xsp)
            init_norm += nplg.norm(init_xsp[:, 0]) ** 2 + (self.hs - 1) * nplg.norm(init_xsp[:, 1]) ** 2
        init_norm = np.sqrt(init_norm)

        norm_trace = list([init_norm])
        # dimensional alternative projected gradient descent optimization
        n_iter = 0
        while (n_iter < max_iters):
            if n_iter % 10 == 0:
                print("iteration: %d" % n_iter)
            xh0_old = deepcopy(xhs[0])
            for dm in range(self.n_dim):
                lbx, ubx = xhs[1][dm], np.ones((self.shape[dm], ))
                xhs[0][dm] = self._projected_optimize_(xh0_old, lbx, ubx, dm, 0, 0, tol=convtol)
                grad_xsdm = self._GRADhierdense_(xhs[0], dm, 0, 0)
                xhs_grad_proj[dm][:, 0] = self._projected_grads_(xhs[0][dm], grad_xsdm, lbx, ubx)
                xh0_old = deepcopy(xhs[0])

            # update for the each hierarchy. i.e. k \in [1, hs - 1]
            for h in range(1, self.hs):
                C = self.etas[h] * self._density_(xhs[h - 1])
                xhs_old = deepcopy(xhs[h])
                # solve the subproblem of xhs[h], and update the dimension alternatively.
                for dm in range(self.n_dim):
                    # lower bound as the current next hierarchy indicator vector, i.e. nested node constraint.
                    lbs_xdm = xhs[h + 1][dm] if h < self.hs - 1 else np.zeros((self.shape[dm], ))
                    # lower bound as the last hierarchy indicator vector.
                    ubs_xdm = xhs[h - 1][dm]
                    xhs[h][dm] = self._projected_optimize_(xhs_old, lbs_xdm, ubs_xdm, dm, h, C, tol=convtol)
                    grad_xsdm_new = self._GRADhierdense_(xhs[h], dm, h, C)
                    xhs_grad_proj[dm][:, h] = self._projected_grads_(xhs[h][dm], grad_xsdm_new, lbs_xdm, ubs_xdm)

            n_iter += 1
            # early stopping (criteria)
            norm_new = 0.0
            for dm in range(self.n_dim):
                norm_new += nplg.norm(xhs_grad_proj[dm]) ** 2
            norm_new = np.sqrt(norm_new)
            norm_trace.append(norm_new)
            if norm_new < eps * init_norm: # or np.abs(norm_trace[-1] - norm_trace[-2]) < convtol:
                break

        if n_iter >= max_iters:
            print("max iterators in nls subproblem")

        if debug:
            print("# iters: {}, norm_init: {}, norm_final: {}".format(n_iter, init_norm, norm_trace[-1]))
            print(norm_trace)
        return xhs

    def _optimal_hiertens_query(self, dims2seeds, cpd_init=None, max_iters=100, eps=1e-7, convtol=1e-6, debug=False):
        '''
        get the optimal hierarchical dense sub-tensors for query
        :param dims2seeds: [dict], the selected seed for specific focus corresponding dimension
        :param cpd_init: [array, None], the initialization for the indicator vector from the tensor CP decomposition
        :param max_iters: [int, default=100], the maximum iterations
        :param eps: [float, default=1e-7], the tolerance threshold
        :param convtol: [float, default=1e-5], the threshold for convergence.
        :return:
        '''
        xhs = list()
        if cpd_init is not None:
            nxs = len(cpd_init)
            xhs.extend(cpd_init[:min([nxs, self.hs])])
            for h in range(self.hs - nxs):
                xhs.append([0.01 * np.ones((dm)) for dm in self.shape])
        else:
            for h in range(self.hs):
                xhs.append([0.01 * np.ones((dm)) for dm in self.shape])
            for dm in range(self.n_dim):
                xhs[0][dm] = 0.5 * np.ones((self.shape[dm]))

        # the average density for the whole tensor by selecting each element
        C0 = self.etas[0] * np.sum(self.T.vals) * 1.0 / np.prod(np.array(self.shape, float))
        # compute the initial gradients for each indicator vector
        grad_x0, grad_xk = list(), list()
        for dm in range(self.n_dim):
            grad_x0.append(self._GRADhierdense_(xhs[0], dm, 0, 0))
            grad_xk.append(self._GRADhierdense_(xhs[1], dm, 1, C0))

        # initial projected gradient norm
        # lower and upper bounds for indicator vectors in the first hierarchy
        lbs = deepcopy(xhs[1])
        for dm, seed in dims2seeds.items():
            lbs[dm][np.array(seed, dtype=int)] = 0.9
        ubs = [np.ones((dm)) for dm in self.shape]

        init_norm = 0.0
        xhs_grad_proj = list()
        for dm in range(self.n_dim):
            init_xsp = np.zeros((self.shape[dm], self.hs))
            init_xsp[:, 0] = self._projected_grads_(xhs[0][dm], grad_x0[dm], lbs[dm], ubs[dm])
            xskp = self._projected_grads_(xhs[1][dm], grad_xk[dm], lbs[dm], ubs[dm])
            init_xsp[:, 1:] = np.repeat(np.array([xskp]), self.hs - 1, 0).T
            xhs_grad_proj.append(init_xsp)
            init_norm += nplg.norm(init_xsp[:, 0]) ** 2 + (self.hs - 1) * nplg.norm(init_xsp[:, 1]) ** 2
        init_norm = np.sqrt(init_norm)

        norm_trace = list([init_norm])
        n_iter = 0
        # dimensional alternative projected gradient descent optimization
        while (n_iter < max_iters):
            if n_iter % 10 == 0:
                print("iteration: %d" % n_iter)
            xh0_old = deepcopy(xhs[0])
            for dm in range(self.n_dim):
                lbx, ubx = copy(xhs[1][dm]), np.ones((self.shape[dm], ))
                if dm in dims2seeds:
                    lbx[np.array(dims2seeds[dm], dtype=int)] = 0.9
                xhs[0][dm] = self._projected_optimize_(xh0_old, lbx, ubx, dm, 0, 0, tol=convtol)
                grad_xsdm = self._GRADhierdense_(xhs[0], dm, 0, 0)
                xhs_grad_proj[dm][:, 0] = self._projected_grads_(xhs[0][dm], grad_xsdm, lbx, ubx)
                xh0_old = copy(xhs[0])

            # update for the each hierarchy. i.e. k \in [1, hs - 1]
            for h in range(1, self.hs):
                C = self.etas[h] * self._density_(xhs[h - 1])
                xhs_old = copy(xhs[h])
                # solve the subproblem of xhs[h], and update the dimension alternatively.
                for dm in range(self.n_dim):
                    # lower bound as the current next hierarchy indicator vector, i.e. nested node constraint.
                    lbs_xdm = copy(xhs[h + 1][dm]) if h < self.hs - 1 else np.zeros((self.shape[dm], ))
                    if dm in dims2seeds:
                        lbs_xdm[np.array(dims2seeds[dm], int)] = 0.9
                    ubs_xdm = copy(xhs[h - 1][dm])
                    xhs[h][dm] = self._projected_optimize_(xhs_old, lbs_xdm, ubs_xdm, dm, h, C, tol=convtol)
                    grad_xsdm_new = self._GRADhierdense_(xhs[h], dm, h, C)
                    xhs_grad_proj[dm][:, h] = self._projected_grads_(xhs[h][dm], grad_xsdm_new, lbs_xdm, ubs_xdm)

            n_iter += 1
            # early stopping (criteria)
            norm_new = 0.0
            for dm in range(self.n_dim):
                norm_new += nplg.norm(xhs_grad_proj[dm]) ** 2
            norm_new = np.sqrt(norm_new)
            norm_trace.append(norm_new)
            if norm_new < eps * init_norm: # or np.abs(norm_trace[-1] - norm_trace[-2]) < convtol:
                break

        if n_iter >= max_iters:
            print("max iterators in nls subproblem")

        if debug:
            print("# iters: {}, norm_init: {}, norm_final: {}".format(n_iter, init_norm, norm_trace[-1]))
            print(norm_trace)
        return xhs

    def hieroptimal(self, max_iters=100, eps=1e-7, convtol=1e-5):
        '''
        hierarchical optimization for dense sub-tensor detection
        :param max_iters: the maximum number of updating iteration
        :param eps: the tolerance value for updating (stop criteria)
        :param convtol: the threshold for convergence.
        :return:
        '''
        return self._optimal_hiertens(max_iters=max_iters, eps=eps, convtol=convtol)

    def queryhiers(self, dims2seeds, cpd_init=None, max_iters=100, eps=1e-7, convtol=1e-5):
        '''
        query specific hierarchical dense sub-densors
        :param dims2seeds: [dict], the selected seed for specific focus corresponding dimension
        :param cpd_init: [array, None], the initialization for the indicator vector from the tensor CP decomposition
        :param max_iters: [int, default=100], the maximum iterations
        :param eps: [float, default=1e-7], the tolerance threshold
        :param convtol: [float, default=1e-5], the threshold for convergence.
        :return:
        '''
        if len(set(dims2seeds.keys()) - set(range(self.n_dim))) > 0:
            ValueError("selected dimension must be in [0, %d]."%(self.n_dim-1))
        return self._optimal_hiertens_query(dims2seeds, cpd_init, max_iters=max_iters, eps=eps, convtol=convtol, debug=True)

    def queryhiers_single(self, seeds, dimension, max_iters=100, eps=1e-7, convtol=1e-5):
        '''
        query specific hierarchical dense sub-tensors
        :param seeds: queried seeds
        :param dimension: selected dimension
        :param max_iters: [int, default=100], the maximum iterations
        :param eps: [float, default=1e-7], the tolerance threshold
        :param convtol: [float, default=1e-5], the threshold for convergence.
        :return:
        '''
        if dimension not in range(self.n_dim):
            ValueError("selected dimension must be in [0, %d]." % (self.n_dim - 1))
        return self._optimal_hiertens_query({dimension: seeds}, max_iters, eps, convtol, debug=True)

    def hierarchy_indicator(self, opt_xs, tholds=0.5):
        '''
        extract the hierarchy indicator vectors from the optimal ordered set opt_xs based on the threshold tholds.
        :param opt_xs: the optimal ordered set from the gradient optimization
        :param tholds: the threshold for indicator vectors
        '''
        n_hiers = len(opt_xs)
        thetas = list()
        hs_valid = list()
        indicators = list()
        hs_nnz, hs_shapes, hs_density = list(), list(), list()

        if isinstance(tholds, collections.Iterable):
           if len(tholds) < self.n_dim:
               thetas = [tholds[0]] * self.n_dim
           else:
               thetas = tholds
        elif tholds is None or not isinstance(tholds, collections.Iterable):
            thetas = [tholds] * self.n_dim

        for h in range(n_hiers):
            print("H{}: {}".format(h + 1, [(np.min(opt_xs[h][dm]), np.max(opt_xs[h][dm])) for dm in range(self.n_dim)]))
            h_idxs, h_shp = list(), list()
            is_diff = 0
            for dm in range(self.n_dim):
                dm_idx = np.where(opt_xs[h][dm] > thetas[dm])[0]
                h_idxs.append(sorted(dm_idx))
                h_shp.append(len(dm_idx))
                if h > 0 and len(indicators) > 0:
                    is_diff += int(set(indicators[-1][dm]) != set(dm_idx))
            cards = np.prod(np.array(h_shp, float))
            if cards > 0 and (h == 0 or (h > 0 and is_diff > 0)):
                nnz, _ = self._subtensor_nnzs_(h_idxs)
                if h == 0 or (h > 0 and nnz != hs_nnz[-1]):
                    hs_valid.append(h)
                    indicators.append(h_idxs)
                    hs_nnz.append(nnz)
                    hs_shapes.append(tuple(h_shp))
                    hs_density.append(nnz * 1.0 / cards)

        self.valid_hs = len(hs_valid)
        self.hs_indicator = indicators

        return hs_valid, indicators, hs_nnz, hs_shapes, hs_density
    
    def _cpds_(self, rank=1):
        '''
        The CP decomposition for the tensor, which can be used as the initialization for the ordered set
        '''
        rkP, fit, _, _ = cp_als(self.T, rank)
        print("tensor decomp res: rank:{}, fit: {}".format(rank, fit))
        hsidxs = list()
        for r in range(rank):
            hdims = list()
            for dm in range(self.n_dim):
                xdm = rkP.U[dm][:, r]
                if np.sum(xdm) < 0:
                    xdm *= -1
                xdm[xdm < 0] = 0.01
                hdims.append(np.array(xdm))
            hsidxs.append(hdims)

        return hsidxs
    
    def dump(self):
        print("Basic Information:")
        print("  n_dims: {}, shape: {}, nnzs: {}, totals: {}".format(self.n_dim, self.shape,
                                                                     len(self.T.vals), np.sum(self.T.vals)))
        print("  entities [min, max]: [{}, {}]".format(np.min(self.T.vals), np.max(self.T.vals)))
        hier_shape_ = ""
        for hs in range(self.valid_hs):
            hshp_str = "(" + ",".join(map(str, [len(self.hs_indicator[hs][dm]) for dm in range(self.n_dim)])) + ")"
            hier_shape_ += hshp_str + ',  '
        print("  Valid hierarchies:{}, shapes:{}".format(self.valid_hs, hier_shape_[:-3]))

        if self.hs_density is not None:
            print("Hierarchical densities and none-zeros (average density: {}):".format(self.hs_density[0]))
            print(" \t{}".format(self.hs_density[1:]))
            print(" \t{}".format(self.hs_nnz))
        print("done!")
