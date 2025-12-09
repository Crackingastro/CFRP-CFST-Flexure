# app.py
import streamlit as st
import glob
import joblib
import os
import pandas as pd
import numpy as np
import random
import optuna
import matplotlib.pyplot as plt
import math
import seaborn as sns
import io

st.set_page_config(
    page_title="CFRP–CFST Flexural Strength Tool",
    layout="wide"
)

# reproducibility
np.random.seed(42)
random.seed(42)
sampler = optuna.samplers.TPESampler(seed=42)

# make relative paths work
if "__file__" in globals():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# CFRP VALIDATION
# ---------------------------------------------------------


def validate_cfrp(fcyd, tc, La, Lc, Ta, Tc):
    if fcyd == 0 and tc == 0:
        if any(x > 0 for x in [La, Lc, Ta, Tc]):
            return False, "CFRP material not defined (fcyd=tc=0), but wrapping parameters are non-zero."
        return True, None

    if (fcyd == 0 and tc > 0) or (fcyd > 0 and tc == 0):
        return False, "CFRP is partially defined: both fcyd and tc must be non-zero."

    long_used = (La > 0) or (Lc > 0)
    if long_used and not (La > 0 and Lc > 0):
        return False, "Longitudinal CFRP is incompletely defined: both La and Lc must be > 0."

    trans_used = (Ta > 0) or (Tc > 0)
    if trans_used and not (Ta > 0 and Tc > 0):
        return False, "Transverse CFRP is incompletely defined: both Ta and Tc must be > 0."

    if not long_used and not trans_used:
        return False, "CFRP is defined (fcyd, tc) but no longitudinal or transverse wrapping was provided."

    return True, None

# ---------------------------------------------------------
# load model
# ---------------------------------------------------------


def load_model(path):
    return joblib.load(path)

# ---------------------------------------------------------
# build pareto df + compromise
# ---------------------------------------------------------


def compute_compromise_and_pareto(MU_all, Mass_all, Const_all, pareto_trials):
    df_pareto = pd.DataFrame([{
        "MU (kNm)": t.values[0],
        "Mass (kg)": t.values[1],
        "Constructability (layers)": t.values[2],
        "params": t.params
    } for t in pareto_trials])

    mu_min, mu_max = min(MU_all), max(MU_all)
    mass_min, mass_max = min(Mass_all), max(Mass_all)
    const_min, const_max = min(Const_all), max(Const_all)

    def norm(x, mn, mx):
        return (x - mn) / (mx - mn) if mx > mn else 0.0

    df_pareto["mu_norm"] = df_pareto["MU (kNm)"].apply(
        lambda x: norm(x, mu_min, mu_max))
    df_pareto["mass_norm"] = df_pareto["Mass (kg)"].apply(
        lambda x: norm(x, mass_min, mass_max))
    df_pareto["const_norm"] = df_pareto["Constructability (layers)"].apply(
        lambda x: norm(x, const_min, const_max))

    df_pareto["dist_to_ideal"] = df_pareto.apply(lambda r: math.sqrt(
        (r.mu_norm - 1.0) ** 2 + r.mass_norm ** 2 + r.const_norm ** 2), axis=1)
    compromise_row = df_pareto.loc[df_pareto["dist_to_ideal"].idxmin()]
    return df_pareto, compromise_row

# ---------------------------------------------------------
# main app
# ---------------------------------------------------------


def main():
    st.title("Flexural Strength Prediction & Optimization — CFRP–CFST")

    # sidebar
    with st.sidebar:
        st.header("Application Settings")
        design_code = st.selectbox(
            "Select design code / region:",
            [
                "Eurocode 4 (EN 1994-1-1)",
                "ACI / AISC + ACI 440.2R",
                "Chinese practice (GB 50936 / CECS 146)",
                "Other / research basis"
            ]
        )
        beta_value = st.number_input(
            "Calibration coefficient β (enter 1.0 if not calibrated):",
            min_value=0.0, max_value=5.0, value=1.0, step=0.01
        )
        st.markdown("**Computation settings**")
        n_trials_opt = st.number_input(
            "MOO trials (n_trials):", 50, 2000, 500, 50)
        n_jobs_opt = st.selectbox(
            "MOO parallelism (n_jobs)", [
                "auto (-1)", "1 (single-thread)"], index=1)
        run_offline_if_heavy = st.checkbox(
            "Prefer offline heavy runs (save/load Pareto)", value=True)

    # model loading
    st.subheader("Model selection")
    model_files = sorted(glob.glob("*.pkl"), reverse=True)
    if not model_files:
        st.error("No .pkl model files found.")
        st.stop()
    selected_model_file = st.selectbox(
        "Select trained model (.pkl):", model_files)
    try:
        model = load_model(selected_model_file)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()
    if not hasattr(model, "gpu_id"):
        model.gpu_id = 0
    if not hasattr(model, "predictor"):
        model.predictor = 0

    st.markdown("---")

    # inputs
    st.subheader("Input Parameters")

    st.markdown("**Steel Properties / Geometry**")
    c1, c2, c3 = st.columns(3)
    with c1:
        d = st.number_input("Diameter / Depth, d (mm):", 0.0, step=1.0)
    with c2:
        b = st.number_input("Width, b (mm):", 0.0, step=1.0)
    with c3:
        ts = st.number_input("Steel thickness, ts (mm):", 0.0, step=1.0)

    c4, c5, _ = st.columns(3)
    with c4:
        fsyd = st.number_input(
            "Steel tensile strength, fsyd (MPa):", 0.0, step=1.0)
    with c5:
        leff = st.number_input("Effective length, leff (mm):", 0.0, step=1.0)

    st.markdown("**Concrete Properties**")
    fcu = st.number_input(
        "Concrete compressive strength, fcu (MPa):",
        0.0,
        step=1.0)

    st.markdown("**CFRP Properties / Configuration**")
    r1 = st.columns(2)
    with r1[0]:
        fcyd = st.number_input(
            "CFRP design strength, fcyd (MPa):", 0.0, step=1.0)
    with r1[1]:
        tc = st.number_input(
            "CFRP thickness per layer, tc (mm):", 0.0, step=0.1)
    r2 = st.columns(2)
    with r2[0]:
        Lc = st.number_input("Longitudinal CFRP layers (Lc):", 0.0, step=1.0)
    with r2[1]:
        La = st.number_input(
            "Longitudinal CFRP coverage (La):", 0.0, step=0.05)
    r3 = st.columns(2)
    with r3[0]:
        Tc_user = st.number_input(
            "Transverse CFRP layers (Tc):", 0.0, step=1.0)
    with r3[1]:
        Ta = st.number_input("Transverse CFRP coverage (Ta):", 0.0, step=0.05)

    columns_order = [
        "d (mm)", "b (mm)", "ts (mm)", "leff (mm)", "fsyd (MPa)",
        "fcu (MPa)", "fcyd (MPa)", "tc (mm)", "LC (No)",
        "La (mm2)", "Tc(No)", "Ta (mm2)"
    ]

    # single prediction
    st.markdown("---")
    st.subheader("Single Prediction")
    if st.button("Predict flexural capacity"):
        if (d == 0 and b == 0) or fsyd == 0 or fcu == 0 or leff == 0:
            st.warning(
                "Please enter non-zero geometry, steel strength, concrete strength, and effective length.")
        else:
            is_valid, err = validate_cfrp(fcyd, tc, La, Lc, Ta, Tc_user)
            if not is_valid:
                st.error(err)
            else:
                df_input = pd.DataFrame(
                    [[d, b, ts, leff, fsyd, fcu, fcyd, tc, Lc, La, Tc_user, Ta]],
                    columns=columns_order
                )
                try:
                    base_pred = model.predict(df_input)[0]
                except Exception as e:
                    st.error(f"Model prediction failed: {e}")
                else:
                    adjusted_pred = base_pred * beta_value
                    st.success(
                        f"Predicted Flexural Capacity = **{adjusted_pred:.3f} kNm**")
                    st.caption(
                        f"(Base = {base_pred:.3f} kNm × β = {beta_value}) — for {design_code}")

# MOO
    st.markdown("---")
    st.subheader(
        "Optimize CFRP Properties (MOO: Max MU, Min Mass, Min Constructability)")

    # REMOVED THE FILE UPLOADER
    pareto_cache_path = "pareto_cached.csv"
    df_pareto = None
    compromise_row = None

# Directly run MOO without upload option
    if st.button("Run MOO (Euclidean compromise)"):

        is_valid, err = validate_cfrp(fcyd, tc, La, Lc, Ta, Tc_user)
        if not is_valid:
            st.error("Cannot start optimization: " + err)

        elif fcyd == 0 or tc == 0:
            st.error(
                "CFRP material properties (fcyd, tc) must be non-zero to run optimization.")

        else:
            MU_ALL, Mass_ALL, Const_ALL = [], [], []

            n_jobs = -1 if n_jobs_opt == "auto (-1)" else 1
            if run_offline_if_heavy and n_jobs == -1 and n_trials_opt > 500:
                st.warning(
                    "You selected many trials with parallel jobs. Consider running offline or reducing trials.")

            with st.spinner("Running multi-objective optimization (Optuna)..."):
                study = optuna.create_study(
                    directions=["maximize", "minimize", "minimize"],
                    sampler=sampler
                )

                def objective(trial):
                    LC_opt = trial.suggest_int("LC", 1, 4)
                    La_opt = trial.suggest_categorical(
                        "La", [0.25, 0.5, 0.75, 1.0])
                    Tc_opt = trial.suggest_int("Tc", 1, 2)
                    Ta_opt = trial.suggest_categorical(
                        "Ta", [0.25, 0.5, 0.75, 1.0])

                    df_input = pd.DataFrame(
                        [[d, b, ts, leff, fsyd, fcu, fcyd, tc,
                          LC_opt, La_opt, Tc_opt, Ta_opt]],
                        columns=columns_order
                    )

                    MU = model.predict(df_input)[0] * beta_value

                    if b == 0:
                        vol_mm3 = np.pi * d * leff * tc * \
                            (LC_opt * La_opt + Tc_opt * Ta_opt)
                    else:
                        vol_mm3 = 2 * (b + d) * leff * tc * \
                            (LC_opt * La_opt + Tc_opt * Ta_opt)

                    mass = vol_mm3 * 1e-9 * 1600
                    constructability = LC_opt + Tc_opt

                    MU_ALL.append(MU)
                    Mass_ALL.append(mass)
                    Const_ALL.append(constructability)

                    return MU, mass, constructability

                study.optimize(
                    objective,
                    n_trials=int(n_trials_opt),
                    n_jobs=n_jobs)
                pareto_trials = study.best_trials
                df_pareto, compromise_row = compute_compromise_and_pareto(
                    MU_ALL, Mass_ALL, Const_ALL, pareto_trials
                )

                try:
                    df_pareto.to_csv(pareto_cache_path, index=False)
                    st.success(
                        f"Optimization complete and Pareto cached to {pareto_cache_path}.")
                except Exception:
                    st.info("Optimization complete (cache write failed).")

    # PARETO PLOT SECTION
    if df_pareto is not None and compromise_row is not None:
        st.subheader("Pareto Visualization")

        df_pareto_plot = df_pareto.copy()
        df_pareto_plot["Compromise"] = False
        df_pareto_plot.loc[
            (df_pareto_plot["MU (kNm)"] == compromise_row["MU (kNm)"]) &
            (df_pareto_plot["Mass (kg)"] == compromise_row["Mass (kg)"]) &
            (df_pareto_plot["Constructability (layers)"] == compromise_row["Constructability (layers)"]),
            "Compromise"
        ] = True

        sns.set(style="ticks")
        g = sns.pairplot(
            df_pareto_plot,
            vars=["MU (kNm)", "Mass (kg)", "Constructability (layers)"],
            diag_kind="kde",
            hue="Compromise",
            palette={False: "blue", True: "red"},
            plot_kws={"alpha": 0.6, "s": 80},
            markers=["o", "o"]
        )
        g.fig.suptitle("A) Square section SBT1L2", y=1.02)
        st.pyplot(g.fig)

        # download button
        buf = io.BytesIO()
        g.fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
        buf.seek(0)
        st.download_button(
            label="📥 Download Plot as PNG",
            data=buf,
            file_name="Pareto_Visualization.png",
            mime="image/png"
        )

        st.markdown("**Compromise solution (closest to ideal):**")
        st.json(compromise_row[["MU (kNm)",
                                "Mass (kg)",
                                "Constructability (layers)",
                                "params"]].to_dict())

    # Monte Carlo
    st.markdown("---")
    st.subheader("Monte Carlo Robustness Check (optional heavy)")
    runs = st.slider(
        "Number of Monte Carlo simulations:",
        10,
        200,
        value=50,
        step=10)
    if st.button("Run Monte Carlo"):
        is_valid, err = validate_cfrp(fcyd, tc, La, Lc, Ta, Tc_user)
        if not is_valid:
            st.error("Cannot run Monte Carlo: " + err)
        else:
            st.info("Running Monte Carlo. This may take some time…")
            mc_mu, mc_mass, mc_const = [], [], []
            for _ in range(runs):
                d_p = np.random.normal(d, 0.05 * d) if d > 0 else d
                b_p = np.random.normal(b, 0.05 * b) if b > 0 else b
                fs_p = np.random.normal(
                    fsyd, 0.10 * fsyd) if fsyd > 0 else fsyd
                le_p = np.random.normal(
                    leff, 0.02 * leff) if leff > 0 else leff

                study = optuna.create_study(
                    directions=["maximize", "minimize", "minimize"],
                    sampler=sampler
                )

                def mc_objective(trial):
                    LC_opt = trial.suggest_int("LC", 1, 4)
                    La_opt = trial.suggest_categorical(
                        "La", [0.25, 0.5, 0.75, 1.0])
                    Tc_opt = trial.suggest_int("Tc", 1, 2)
                    Ta_opt = trial.suggest_categorical(
                        "Ta", [0.25, 0.5, 0.75, 1.0])
                    df_in = pd.DataFrame(
                        [[d_p, b_p, ts, le_p, fs_p, fcu, fcyd, tc,
                          LC_opt, La_opt, Tc_opt, Ta_opt]],
                        columns=columns_order
                    )
                    MU = model.predict(df_in)[0] * beta_value
                    if b == 0:
                        vol_mm3 = np.pi * d_p * le_p * tc * \
                            (LC_opt * La_opt + Tc_opt * Ta_opt)
                    else:
                        vol_mm3 = 2 * (b_p + d_p) * le_p * tc * \
                            (LC_opt * La_opt + Tc_opt * Ta_opt)
                    mass = vol_mm3 * 1e-9 * 1600
                    constructability = LC_opt + Tc_opt
                    return MU, mass, constructability

                study.optimize(mc_objective, n_trials=40, n_jobs=1)
                for t in study.best_trials:
                    mu, m, c = t.values
                    mc_mu.append(mu)
                    mc_mass.append(m)
                    mc_const.append(c)

            df_mc = pd.DataFrame({
                "MU (kNm)": mc_mu,
                "Mass (kg)": mc_mass,
                "Constructability (layers)": mc_const
            })

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.kdeplot(
                x=df_mc["Mass (kg)"],
                y=df_mc["MU (kNm)"],
                fill=True,
                cmap="Blues",
                levels=15,
                alpha=0.3,
                ax=ax
            )
            sc = ax.scatter(
                df_mc["Mass (kg)"],
                df_mc["MU (kNm)"],
                c=df_mc["Constructability (layers)"],
                cmap="viridis",
                alpha=0.6
            )
            plt.colorbar(sc, ax=ax, label="Constructability (layers)")
            ax.set_xlabel("Mass (kg)")
            ax.set_ylabel("MU (kNm)")
            st.pyplot(fig)

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=300, bbox_inches="tight")
            buf.seek(0)
            st.download_button(
                "Download Monte Carlo figure (PNG)",
                data=buf,
                file_name="mc_pareto_cloud.png",
                mime="image/png"
            )


if __name__ == "__main__":
    main()
