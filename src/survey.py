"""Design-based (Taylor linearization) survey estimation for stratified multistage designs.
Matches R 'survey' package svyglm / svymean behaviour with with-replacement PSU variance.
"""
import numpy as np, pandas as pd
from scipy import stats as st

def _psu_totals(u, strat, psu):
    """Aggregate score matrix u (n x p) to PSU totals; return dict stratum -> array(n_h x p)."""
    df = pd.DataFrame(u)
    df['_h'] = np.asarray(strat); df['_i'] = np.asarray(psu)
    g = df.groupby(['_h', '_i'], sort=True).sum()
    out = {}
    for h, sub in g.groupby(level=0):
        out[h] = sub.to_numpy()
    return out

def design_vcov(u, strat, psu):
    """V = sum_h n_h/(n_h-1) sum_i (u_hi - ubar_h)(.)' ; strata with 1 PSU contribute 0."""
    p = u.shape[1]
    V = np.zeros((p, p))
    nps, nstr = 0, 0
    for h, U in _psu_totals(u, strat, psu).items():
        n_h = U.shape[0]; nps += n_h; nstr += 1
        if n_h < 2:
            continue
        d = U - U.mean(axis=0, keepdims=True)
        V += (n_h / (n_h - 1.0)) * (d.T @ d)
    return V, nps - nstr

class SvyDesign:
    def __init__(self, data, weight, strata, psu):
        self.d = data.reset_index(drop=True)
        self.w = self.d[weight].to_numpy(float)
        self.h = self.d[strata].to_numpy()
        self.p = pd.Series(list(zip(self.d[strata], self.d[psu]))).to_numpy()
        # degrees of freedom for full design
        t = self.d[[strata, psu]].drop_duplicates()
        self.df = t.shape[0] - self.d[strata].nunique()

    def subset(self, mask):
        """Domain estimation: keep full design, zero-weight the excluded (correct SEs)."""
        s = SvyDesign.__new__(SvyDesign)
        s.d = self.d; s.w = np.where(np.asarray(mask), self.w, 0.0)
        s.h = self.h; s.p = self.p; s.df = self.df
        return s

    def mean(self, y, ci=True):
        y = np.asarray(y, float); w = self.w
        ok = ~np.isnan(y)
        w = np.where(ok, w, 0.0); yy = np.where(ok, y, 0.0)
        W = w.sum(); mu = (w * yy).sum() / W
        u = (w * (yy - mu))[:, None]          # linearized ratio residual
        V, dfree = design_vcov(u, self.h, self.p)
        se = np.sqrt(V[0, 0]) / W
        if not ci: return mu, se
        tq = st.t.ppf(0.975, max(self.df, 1))
        return mu, se, mu - tq * se, mu + tq * se

    def prop(self, y):
        """Proportion with logit (Korn-Graubard style) CI on logit scale."""
        mu, se = self.mean(y, ci=False)
        tq = st.t.ppf(0.975, max(self.df, 1))
        if 0 < mu < 1:
            lg = np.log(mu / (1 - mu)); lse = se / (mu * (1 - mu))
            lo = 1 / (1 + np.exp(-(lg - tq * lse))); hi = 1 / (1 + np.exp(-(lg + tq * lse)))
        else:
            lo, hi = max(0.0, mu - tq * se), min(1.0, mu + tq * se)
        n_eff = int((~np.isnan(np.asarray(y, float)) & (self.w > 0)).sum())
        return dict(est=mu, se=se, lo=lo, hi=hi, n=n_eff,
                    N=float((self.w * np.nan_to_num(np.asarray(y, float))).sum()))

    def total(self, y):
        y = np.asarray(y, float); ok = ~np.isnan(y)
        w = np.where(ok, self.w, 0.0); yy = np.where(ok, y, 0.0)
        T = (w * yy).sum(); u = (w * yy)[:, None]
        V, _ = design_vcov(u, self.h, self.p)
        return T, np.sqrt(V[0, 0])

def _irls_logit(X, y, w, maxit=200, tol=1e-9, ridge=1e-8):
    """Newton-Raphson with step-halving; tiny ridge for numerical stability."""
    b = np.zeros(X.shape[1])

    def dev(bb):
        eta = np.clip(X @ bb, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        mu = np.clip(mu, 1e-12, 1 - 1e-12)
        return -2 * np.sum(w * (y * np.log(mu) + (1 - y) * np.log(1 - mu)))

    d0 = dev(b)
    R = ridge * np.eye(X.shape[1]); R[0, 0] = 0.0
    for _ in range(maxit):
        eta = np.clip(X @ b, -30, 30)
        mu = 1 / (1 + np.exp(-eta))
        v = np.clip(mu * (1 - mu), 1e-10, None)
        W = w * v
        XtWX = X.T @ (X * W[:, None]) + R
        z = X.T @ (w * (y - mu)) - R @ b
        try:
            step = np.linalg.solve(XtWX, z)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(XtWX) @ z
        t = 1.0
        for _h in range(30):                       # step-halving on the deviance
            bn = b + t * step
            dn = dev(bn)
            if np.isfinite(dn) and dn <= d0 + 1e-8:
                break
            t *= 0.5
        else:
            break
        moved = np.max(np.abs(bn - b))
        b, d0 = bn, dn
        if moved < tol:
            break
    eta = np.clip(X @ b, -30, 30)
    mu = 1 / (1 + np.exp(-eta))
    return b, mu


def _drop_constant(X, names, w):
    """Drop columns that are constant or all-zero within the estimation sample
    (e.g. a year dummy in a single-year subset), keeping the intercept."""
    keep = [0]
    act = w > 0
    for j in range(1, X.shape[1]):
        col = X[act, j]
        if np.nanstd(col) > 1e-12:
            keep.append(j)
    return X[:, keep], [names[j] for j in keep], keep


def svylogit(design, formula_y, X, mask=None, names=None):
    """Weighted logistic regression with design-based (linearized) covariance."""
    w = design.w.copy()
    y = np.asarray(formula_y, float)
    keep = ~np.isnan(y) & ~np.isnan(X).any(axis=1) & (w > 0)
    if mask is not None: keep &= np.asarray(mask)
    wv = np.where(keep, w, 0.0)
    Xv = np.nan_to_num(X); yv = np.nan_to_num(y)
    Xv, names_k, keep_idx = _drop_constant(Xv, list(names or [f'x{i}' for i in range(X.shape[1])]), wv)
    names = names_k
    b, mu = _irls_logit(Xv, yv, wv)
    resid = np.where(keep, yv - mu, 0.0)
    u = (wv * resid)[:, None] * Xv                      # score contributions
    Vscore, _ = design_vcov(u, design.h, design.p)
    D = Xv.T @ (Xv * (wv * mu * (1 - mu))[:, None])     # weighted information
    Dinv = np.linalg.pinv(D)
    V = Dinv @ Vscore @ Dinv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    dfree = design.df
    tq = st.t.ppf(0.975, max(dfree, 1))
    t = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
    p = 2 * st.t.sf(np.abs(t), max(dfree, 1))
    return pd.DataFrame(dict(term=names, beta=b, se=se, OR=np.exp(np.clip(b, -30, 30)),
                             lo=np.exp(np.clip(b - tq * se, -30, 30)),
                             hi=np.exp(np.clip(b + tq * se, -30, 30)),
                             t=t, p=p)), V, int(keep.sum()), dfree

def svylin(design, yv, X, mask=None, names=None):
    w = design.w.copy(); y = np.asarray(yv, float)
    keep = ~np.isnan(y) & ~np.isnan(X).any(axis=1) & (w > 0)
    if mask is not None: keep &= np.asarray(mask)
    wv = np.where(keep, w, 0.0); Xv = np.nan_to_num(X); yy = np.nan_to_num(y)
    Xv, names_k, keep_idx = _drop_constant(Xv, list(names or [f'x{i}' for i in range(X.shape[1])]), wv)
    names = names_k
    XtWX = Xv.T @ (Xv * wv[:, None])
    b = np.linalg.pinv(XtWX) @ (Xv.T @ (wv * yy))
    r = np.where(keep, yy - Xv @ b, 0.0)
    u = (wv * r)[:, None] * Xv
    Vs, _ = design_vcov(u, design.h, design.p)
    Dinv = np.linalg.pinv(XtWX)
    V = Dinv @ Vs @ Dinv
    se = np.sqrt(np.clip(np.diag(V), 0, None))
    tq = st.t.ppf(0.975, max(design.df, 1))
    t = np.divide(b, se, out=np.zeros_like(b), where=se > 0)
    p = 2 * st.t.sf(np.abs(t), max(design.df, 1))
    return pd.DataFrame(dict(term=names, beta=b, se=se, lo=b - tq * se, hi=b + tq * se,
                             t=t, p=p)), V, int(keep.sum()), design.df

def wald(b, V, idx, dfree):
    """Multi-df Wald test (F form) for a set of coefficient indices."""
    idx = list(idx); k = len(idx)
    bb = np.asarray(b)[idx]; VV = np.asarray(V)[np.ix_(idx, idx)]
    chi2 = bb @ np.linalg.pinv(VV) @ bb
    F = chi2 / k
    return F, k, dfree - k + 1, st.f.sf(F, k, max(dfree - k + 1, 1))

def contrast(b, V, cvec, dfree):
    c = np.asarray(cvec, float); est = float(c @ b); se = float(np.sqrt(c @ V @ c))
    tq = st.t.ppf(0.975, max(dfree, 1))
    return est, se, est - tq * se, est + tq * se, 2 * st.t.sf(abs(est / se), max(dfree, 1))
