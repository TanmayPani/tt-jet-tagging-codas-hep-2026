import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    import matplotlib.pyplot as plt
    import seaborn as sns

    from jet_util import load_images, pixelize, fetch_data_path
    from jet_dataloader import prepare_constituents
    from rocket import RocketTransform2D
    from xgboost import XGBClassifier
    from sklearn.metrics import roc_auc_score, roc_curve, auc, confusion_matrix

    return (
        RocketTransform2D,
        XGBClassifier,
        fetch_data_path,
        load_images,
        mo,
        np,
        pixelize,
        pl,
        plt,
        prepare_constituents,
        roc_auc_score,
        roc_curve,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # QCD vs. top tagging — 2D-ROCKET features + XGBoost

    Pipeline: **training images → augment (pT/position smear) → canonicalize →
    2D-ROCKET featurize → XGBoost → evaluate on validation.**

    The ROCKET transform is frozen/random; its only *fitted* piece is the
    per-kernel ppv bias, calibrated on the (augmented) training images. XGBoost
    does all the actual learning on top of the fixed features.
    """)
    return


@app.cell
def _(mo):
    # Script mode (`uv run rocket_xgb_tagger.py`) uses small caps so the whole
    # pipeline runs end-to-end quickly; interactive mode uses the slider values.
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell
def _(mo):
    seed = mo.ui.number(0, 9999, value=0, label="seed")
    n_kernels = mo.ui.number(500, 10000, step=500, value=10000, label="ROCKET kernels")
    n_biases = mo.ui.number(1, 6, step=1, value=3, label="ppv biases / kernel")
    pt_smear = mo.ui.number(0.0, 0.2, step=0.01, value=0.05, label="pT smear sigma")
    pos_smear = mo.ui.number(0.0, 0.2, step=0.01, value=0.05, label="position smear sigma")
    n_aug = mo.ui.number(0, 6, step=1, value=4, label="augmented copies / train jet")
    l1_norm = mo.ui.checkbox(value=False, label="L1-normalize images (drops energy scale)")
    n_estimators = mo.ui.number(50, 1200, step=50, value=400, label="XGB trees")
    max_depth = mo.ui.number(2, 10, step=1, value=5, label="XGB max_depth")
    learning_rate = mo.ui.number(0.01, 0.3, step=0.01, value=0.05, label="XGB learning_rate")

    mo.vstack(
        [
            mo.md("### Controls"),
            seed,
            n_kernels,
            n_biases,
            pt_smear,
            pos_smear,
            n_aug,
            l1_norm,
            n_estimators,
            max_depth,
            learning_rate,
        ]
    )
    return (
        l1_norm,
        learning_rate,
        max_depth,
        n_aug,
        n_biases,
        n_estimators,
        n_kernels,
        pos_smear,
        pt_smear,
        seed,
    )


@app.cell
def _(load_images):
    (X_train, y_train, _train_ids, X_val, y_val, _val_ids, X_test, test_ids) = load_images()
    return X_test, X_train, X_val, test_ids, y_train, y_val


@app.cell
def _(np, pixelize, prepare_constituents):
    def build_image_array(
        images, augment, n_copies, pt_smear, pos_smear, seed, cap=None, normalize=False
    ):
        """Turn raw jet images into canonicalized 30x30 images.

        When ``augment`` is set, each jet contributes one clean copy plus
        ``n_copies`` smeared copies (smear-then-canonicalize, matching the training
        pipeline). ``normalize`` L1-normalizes each image to sum 1; leaving it off
        keeps the (max-normalized) energy scale, which ROCKET can then exploit.
        Returns the image stack and the index of the source jet for each row, so
        labels can be gathered with ``labels[label_idx]``.
        """
        if cap is not None:
            images = images[:cap]
        rng = np.random.default_rng(seed)
        grids, label_idx = [], []
        for i, img in enumerate(images):
            variants = 1 + (n_copies if augment else 0)
            for v in range(variants):
                do_aug = augment and v > 0  # v == 0 is the clean copy
                cons = prepare_constituents(
                    img,
                    augment=do_aug,
                    canonicalize=True,
                    pt_smear=pt_smear,
                    pos_smear=pos_smear,
                    rng=rng,
                )
                grid = pixelize(cons).astype(np.float32)
                if normalize:
                    total = grid.sum()
                    if total > 0:
                        grid = grid / total
                grids.append(grid)
                label_idx.append(i)
        return np.stack(grids), np.asarray(label_idx)

    return (build_image_array,)


@app.cell
def _(
    X_test,
    X_train,
    X_val,
    build_image_array,
    is_script_mode,
    l1_norm,
    n_aug,
    pos_smear,
    pt_smear,
    seed,
    y_train,
    y_val,
):
    _train_cap = 400 if is_script_mode else None
    _val_cap = 200 if is_script_mode else None

    # Train: augmented (clean + smeared copies). Val/test: clean only.
    Xtr_img, tr_idx = build_image_array(
        X_train,
        augment=True,
        n_copies=n_aug.value,
        pt_smear=pt_smear.value,
        pos_smear=pos_smear.value,
        seed=seed.value,
        cap=_train_cap,
        normalize=l1_norm.value,
    )
    ytr = y_train[tr_idx].astype(int)

    Xval_img, _val_idx = build_image_array(
        X_val,
        augment=False,
        n_copies=0,
        pt_smear=0.0,
        pos_smear=0.0,
        seed=seed.value,
        cap=_val_cap,
        normalize=l1_norm.value,
    )
    yval = y_val[_val_idx].astype(int)

    # Test: unlabelled, always the full set so the submission covers every test id.
    Xtest_img, _ = build_image_array(
        X_test,
        augment=False,
        n_copies=0,
        pt_smear=0.0,
        pos_smear=0.0,
        seed=seed.value,
        normalize=l1_norm.value,
    )
    return Xtest_img, Xtr_img, Xval_img, tr_idx, ytr, yval


@app.cell(hide_code=True)
def _(Xtr_img, Xval_img, mo, ytr, yval):
    mo.md(f"""
    **Train images** (after augmentation): `{Xtr_img.shape}` —
    class balance {ytr.mean():.2f} top.
    **Val images**: `{Xval_img.shape}` — class balance {yval.mean():.2f} top.
    """)
    return


@app.cell
def _(
    RocketTransform2D,
    Xtest_img,
    Xtr_img,
    Xval_img,
    is_script_mode,
    n_biases,
    n_kernels,
    np,
    seed,
):
    import gc as _gc
    import torch as _torch

    _k = 500 if is_script_mode else n_kernels.value
    _rocket = RocketTransform2D(n_kernels=_k, n_ppv_biases=n_biases.value, seed=seed.value)

    # Calibrate the ppv biases on a random training subsample (quantiles are stable
    # from a few thousand jets), then featurize every image.
    _rng = np.random.default_rng(seed.value)
    _fit_n = min(1500, len(Xtr_img))
    _fit_idx = _rng.choice(len(Xtr_img), size=_fit_n, replace=False)
    _rocket.fit(Xtr_img[_fit_idx])

    # Cache the features to disk, then drop the GPU-resident ROCKET module: there is
    # no reason to keep the featurizer on the GPU while XGBoost trains (also on GPU).
    feature_cache = "rocket_features.npz"
    np.savez(
        feature_cache,
        Ftr=_rocket.transform(Xtr_img),
        Fval=_rocket.transform(Xval_img),
        Ftest=_rocket.transform(Xtest_img),
    )
    del _rocket
    _gc.collect()
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    return (feature_cache,)


@app.cell
def load_features(feature_cache, np):
    # Features come back from disk, so the ROCKET module is already off the GPU.
    _features = np.load(feature_cache)
    Ftr = _features["Ftr"]
    Fval = _features["Fval"]
    Ftest = _features["Ftest"]
    return Ftest, Ftr, Fval


@app.cell
def physics_features(fetch_data_path, np, pl):
    # Shipped per-jet physics features (energy scale, cluster / mass proxy, multiplicity),
    # row-aligned to the images. Drop zero-variance columns (e.g. max_energy is const 1.0).
    def _load_phys(split):
        _dir = fetch_data_path() / split / "features"
        _raw = pl.read_csv(_dir / f"raw_features_{split}.csv")
        _clu = pl.read_csv(_dir / "cluster_features.csv")
        return _raw.hstack(_clu).to_numpy().astype(np.float32)

    _ptr = _load_phys("train")
    _keep = _ptr.std(axis=0) > 0
    phys_train = _ptr[:, _keep]
    phys_val = _load_phys("val")[:, _keep]
    phys_test = _load_phys("test")[:, _keep]
    return phys_test, phys_train, phys_val


@app.cell
def combine_features(
    Ftest,
    Ftr,
    Fval,
    np,
    phys_test,
    phys_train,
    phys_val,
    tr_idx,
):
    # ROCKET (image texture) + shipped physics (energy / mass / multiplicity). Broadcast
    # each jet\'s physics row to all of its augmented training copies via tr_idx.
    Ftr_all = np.hstack([Ftr, phys_train[tr_idx]])
    Fval_all = np.hstack([Fval, phys_val])
    Ftest_all = np.hstack([Ftest, phys_test])

    # Feature routes to compare -- each trains its own model and writes its own
    # solutions_<route>.csv, so they can be submitted independently.
    routes = {
        "rocket": (Ftr, Fval, Ftest),
        "physics": (phys_train[tr_idx], phys_val, phys_test),
        "combined": (Ftr_all, Fval_all, Ftest_all),
    }
    return Ftr_all, Fval_all, routes


@app.cell(hide_code=True)
def _(Ftr, Ftr_all, Fval_all, mo, phys_train):
    mo.md(f"""
    **Features** — ROCKET `{Ftr.shape[1]}` + physics `{phys_train.shape[1]}` = `{Ftr_all.shape[1]}` per jet.
    Train `{Ftr_all.shape}`, val `{Fval_all.shape}`.
    """)
    return


@app.cell
def _(
    XGBClassifier,
    is_script_mode,
    learning_rate,
    max_depth,
    n_estimators,
    pl,
    roc_auc_score,
    routes,
    seed,
    test_ids,
    ytr,
    yval,
):
    _trees = 60 if is_script_mode else n_estimators.value
    # scale_pos_weight = n_neg / n_pos handles the ~1.9:1 QCD:top imbalance.
    _scale_pos_weight = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    _params = dict(
        n_estimators=_trees,
        max_depth=max_depth.value,
        learning_rate=learning_rate.value,
        subsample=0.8,
        colsample_bytree=0.3,
        scale_pos_weight=_scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        random_state=seed.value,
        n_jobs=-1,
    )

    def _fit(Xt):
        # prefer GPU, fall back to CPU on OOM (the wide ROCKET matrices exceed 8 GB)
        try:
            c = XGBClassifier(device="cuda", **_params)
            c.fit(Xt, ytr)
        except Exception:
            c = XGBClassifier(device="cpu", **_params)
            c.fit(Xt, ytr)
        return c

    # Train one model per route; write a separate submission CSV for each.
    results = {}
    for _name, (_Xt, _Xv, _Xte) in routes.items():
        _clf = _fit(_Xt)
        _val_p = _clf.predict_proba(_Xv)[:, 1]
        _path = f"solutions_{_name}.csv"
        pl.DataFrame({"id": test_ids, "label": _clf.predict_proba(_Xte)[:, 1]}).write_csv(_path)
        results[_name] = {
            "val_proba": _val_p,
            "auc": float(roc_auc_score(yval, _val_p)),
            "path": _path,
        }
    return (results,)


@app.cell
def _(mo, plt, results, roc_curve, yval):
    _fig, _ax = plt.subplots(figsize=(5, 5))
    for _name, _r in results.items():
        _fpr, _tpr, _ = roc_curve(yval, _r["val_proba"])
        _ax.plot(_fpr, _tpr, label=f"{_name}: {_r['auc']:.4f}")
    _ax.plot([0, 1], [0, 1], "--", color="grey")
    _ax.set_xlabel("False positive rate")
    _ax.set_ylabel("True positive rate")
    _ax.set_title("ROC — validation, by route")
    _ax.legend(loc="lower right")

    _rows = "\n".join(
        f"| {_name} | {_r['auc']:.4f} | `{_r['path']}` |" for _name, _r in results.items()
    )
    mo.vstack(
        [
            mo.md("### Validation AUC by route\n\n| route | AUC | file |\n|---|---|---|\n" + _rows),
            _fig,
        ]
    )
    return


if __name__ == "__main__":
    app.run()
