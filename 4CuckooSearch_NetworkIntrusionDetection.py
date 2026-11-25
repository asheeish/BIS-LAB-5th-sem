import numpy as np
from math import gamma
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def levy(beta, dim):
    """Levy step sizes (Mantegna algorithm)."""
    s = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
         (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    return np.random.normal(0, s, dim) / np.abs(np.random.normal(0, 1, dim)) ** (1 / beta)

def to_mask(pos):
    """Continuous position -> binary feature mask (sigmoid > 0.5 is the same as pos > 0)."""
    m = pos > 0
    if not m.any():
        m[np.random.randint(len(m))] = True
    return m

def rf_acc(m, data):
    X_tr, X_te, y_tr, y_te = data
    clf = RandomForestClassifier(n_estimators=30, random_state=42).fit(X_tr[:, m], y_tr)
    return accuracy_score(y_te, clf.predict(X_te[:, m]))

def fitness(pos, data, w_err=0.9, w_feat=0.1):
    m = to_mask(pos)
    return w_err * (1 - rf_acc(m, data)) + w_feat * m.mean()

def cuckoo_search(data, nests_n=15, p_a=0.25, iters=20, alpha=0.01):
    d = data[0].shape[1]
    rand = lambda *shape: np.random.uniform(-2, 2, shape)
    nests = rand(nests_n, d)
    fit = np.array([fitness(n, data) for n in nests])
    best, best_fit = nests[fit.argmin()].copy(), fit.min()

    for it in range(iters):
        for i in range(nests_n):
            new = nests[i] + alpha * levy(1.5, d)
            f, j = fitness(new, data), np.random.randint(nests_n)
            if f < fit[j]:
                nests[j], fit[j] = new, f

        order = fit.argsort()
        nests, fit = nests[order], fit[order]

        for k in range(nests_n - int(p_a * nests_n), nests_n):  # abandon worst nests
            nests[k] = rand(d)
            fit[k] = fitness(nests[k], data)

        order = fit.argsort()
        nests, fit = nests[order], fit[order]

        if fit[0] < best_fit:
            best, best_fit = nests[0].copy(), fit[0]
        print(f"Iteration {it + 1}/{iters} - Best Fitness: {best_fit:.4f}")
    return best, best_fit

if __name__ == "__main__":
    names = ["duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
             "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
             "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
             "num_shells", "num_access_files", "count", "srv_count", "serror_rate",
             "rerror_rate", "same_srv_rate", "diff_srv_rate"]

    X, y = make_classification(n_samples=600, n_features=25, n_informative=12,
                               n_redundant=8, random_state=42)
    data = train_test_split(X, y, test_size=0.3, random_state=42)

    best, _ = cuckoo_search(data, nests_n=12, iters=10)
    mask = to_mask(best)

    print("\n=========== Optimization Results ===========")
    print(f"Total Original Features : {X.shape[1]}")
    print(f"Baseline Accuracy (All) : {rf_acc(np.ones(X.shape[1], bool), data) * 100:.2f}%")
    print(f"Optimized Feature Count : {mask.sum()}")
    print(f"Optimized Accuracy      : {rf_acc(mask, data) * 100:.2f}%")
    print(f"Selected Features       : {[n for n, m in zip(names, mask) if m]}")