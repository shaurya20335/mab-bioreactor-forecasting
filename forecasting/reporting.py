"""Rebuild publication figures and a readable report from saved experiment artifacts."""

import json
import os
from pathlib import Path
import textwrap

import numpy as np

from bioreactor import Operation, simulate
from .serialization import load_mab_results


LABELS = {"persistence": "Persistence", "linear_drift": "Linear drift", "quadratic": "Quadratic",
          "ridge": "Ridge + cubic correction", "mlp_ensemble": "MLP ensemble", "lstm_ensemble": "LSTM ensemble"}


def build_report(output_dir="results/final"):
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out = Path(output_dir)
    metrics = load_mab_results(out / "metrics.json")
    selection = load_mab_results(out / "selection.json")
    manifest = json.loads((out / "data/manifest.json").read_text())
    design, selected = metrics["design"], metrics["selected_model"]
    data = np.load(out / "plot_data.npz", allow_pickle=False)
    names = [name for name in LABELS if name in metrics["test"]]
    # Show the best individual neural candidates selected on validation as well.
    for family in ("mlp", "lstm"):
        seed_names = [name for name in metrics["validation"] if name.startswith(family+"_seed")]
        if seed_names:
            name = min(seed_names, key=lambda n: metrics["validation"][n]["rmse_mg_L"])
            names.append(name)
    names.sort(key=lambda n: metrics["validation"][n]["rmse_mg_L"])
    test_score, future_score = metrics["test"][selected], metrics["future_time_test"][selected]
    best_lstm = min((name for name in metrics["validation"] if name.startswith("lstm")),
                    key=lambda n: metrics["validation"][n]["rmse_mg_L"])
    radius = metrics["intervals"]["radius_mg_L"]
    leads = np.arange(1, design["horizon"]+1)*design["step_min"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    positions = np.arange(len(names))
    labels = [LABELS.get(name, name.replace("_", " ")) for name in names]
    for ax, split, title in zip(axes, ("validation", "test"), ("Validation: model selection", "Unseen batches: final test")):
        scores = [metrics[split][name]["rmse_mg_L"] for name in names]
        ax.barh(positions, scores, color=["#14806B" if n == selected else "#657E9B" for n in names])
        ax.set_yticks(positions, labels)
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set(xlabel="RMSE across all leads (mg/L, log scale)", title=title)
        ax.grid(axis="x", alpha=.2)
    fig.savefig(out / "model_comparison.png", dpi=180)
    fig.savefig(out / "model_comparison.pdf")
    comparison_fig = fig

    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.axis("off")
    cells = [[LABELS.get(name, name), *[f"{metrics[split][name]['rmse_mg_L']:.6g}"
             for split in ("validation", "test", "future_time_test")]] for name in names]
    table_artist = ax.table(cellText=cells,
                            colLabels=["Model", "Validation", "Unseen batches", "Later-time test"],
                            colWidths=[.4, .2, .2, .2], loc="center", cellLoc="left")
    table_artist.auto_set_font_size(False)
    table_artist.set_fontsize(11)
    table_artist.scale(1, 1.8)
    for (row, col), cell in table_artist.get_celld().items():
        cell.set_edgecolor("#DBE2E8")
        if row == 0:
            cell.set_facecolor("#25384B")
            cell.get_text().set_color("white")
        elif names[row-1] == selected:
            cell.set_facecolor("#D9F1EA")
    ax.set_title("Forecast RMSE (mg/L) | selected model highlighted", fontsize=15, pad=20)
    fig.savefig(out / "results_table.png", dpi=180, bbox_inches="tight", pad_inches=.2)
    fig.savefig(out / "results_table.pdf", bbox_inches="tight", pad_inches=.2)
    table_fig = fig

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for ax, mode in zip(axes.flat, ("batch", "fed_batch", "continuous", "retained_perfusion")):
        idx = np.flatnonzero(data["future_modes"] == mode)[0]
        history = data["future_history"][idx, :, 2]
        origin = design["development_end_min"]
        history_times = origin + np.arange(1-design["window"], 1)*design["step_min"]
        pred = data["future_prediction"][idx]
        ax.plot(history_times, history, color="#304356", label="Observed simulated history")
        ax.plot(origin+leads, data["future_actual"][idx], color="#304356", linestyle="--", label="Held-out simulator values")
        ax.plot(origin+leads, pred, color="#16806B", label="Selected model")
        ax.fill_between(origin+leads, np.maximum(0, pred-radius), pred+radius,
                        color="#16806B", alpha=.18, label="90% nominal batch band")
        ax.axvline(origin, color="gray", linestyle=":")
        ax.set(title=mode.replace("_", " ").title(), xlabel="Time (min)", ylabel="mAb (mg/L)")
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Forecasts on unseen batches beyond the training time range", fontsize=14)
    fig.savefig(out / "heldout_forecasts.png", dpi=180)
    fig.savefig(out / "heldout_forecasts.pdf")
    heldout_fig = fig

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    error = data["test_prediction"]-data["test_actual"]
    later_error = data["future_prediction"]-data["future_actual"]
    axes[0].plot(leads, np.sqrt(np.mean(error**2, axis=0)), label="Unseen batches, development time")
    axes[0].plot(leads, np.sqrt(np.mean(later_error**2, axis=0)), label="Unseen batches, later time")
    axes[0].set(xlabel="Forecast lead (min)", ylabel="RMSE (mg/L)", title="Error increases with forecast horizon")
    axes[0].legend(fontsize=8)
    clean = [metrics["test"][name]["rmse_mg_L"] for name in names]
    noisy = [metrics["measurement_noise_sensitivity"]["scores"][name]["rmse_mg_L"] for name in names]
    axes[1].barh(positions-.18, clean, height=.35, label="Clean synthetic history", color="#16806B")
    axes[1].barh(positions+.18, noisy, height=.35, label="Perturbed history", color="#C58042")
    axes[1].set_yticks(positions, labels)
    axes[1].invert_yaxis()
    axes[1].set_xscale("log")
    axes[1].set(xlabel="RMSE (mg/L, log scale)", title="Measurement-noise sensitivity")
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(alpha=.2)
    fig.savefig(out / "error_and_noise.png", dpi=180)
    fig.savefig(out / "error_and_noise.pdf")
    sensitivity_fig = fig

    project = np.loadtxt(out / "project_forecast.csv", delimiter=",", skiprows=1)
    project_metadata = json.loads((out / "project_forecast.json").read_text())
    raw = np.loadtxt(project_metadata["history"], delimiter=",", skiprows=1)
    raw = raw[raw[:, 0] <= project_metadata["origin_min"]]
    # Independent reference continuation is ONLY an illustration after selection;
    # the inference command itself neither calls simulate nor sees future data.
    initial = raw[-1, 1:].copy()
    initial[5] = initial[1]*(100/initial[5]-1) if initial[5] > 0 else 0
    reference = simulate(np.r_[project_metadata["origin_min"], project[:, 0]], initial,
                         operation=Operation(**project_metadata["operation"]))[1:, 2]
    project_score = {"rmse_mg_L": float(np.sqrt(np.mean((project[:, 1]-reference)**2))),
                     "final_forecast_mg_L": float(project[-1, 1]),
                     "final_simulator_reference_mg_L": float(reference[-1]),
                     "role": "Illustrative default-batch continuation after model selection, not an independent experiment."}
    (out / "project_example_metrics.json").write_text(json.dumps(project_score, indent=2)+"\n")
    np.savetxt(out / "project_reference.csv", np.column_stack((project[:, 0], reference)),
               delimiter=",", comments="", header="time_min,simulator_reference_mab_mg_L")
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
    ax.plot(raw[:, 0], raw[:, 3], label="Project history", color="#304356")
    ax.plot(project[:, 0], reference, "--", label="Independent ODE continuation", color="#304356")
    ax.plot(project[:, 0], project[:, 1], label="Selected forecast", color="#16806B")
    ax.fill_between(project[:, 0], project[:, 2], project[:, 3], color="#16806B", alpha=.2,
                    label="90% nominal synthetic batch band")
    ax.axvline(raw[-1, 0], linestyle=":", color="gray")
    ax.set(xlabel="Time (min)", ylabel="mAb (mg/L)", title="Forecast from the project's 5,000-minute observation boundary")
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(out / "project_forecast.png", dpi=180)
    fig.savefig(out / "project_forecast.pdf")
    project_fig = fig

    table = ["| Model | Validation RMSE | Test RMSE | Later-time test RMSE |",
             "| --- | ---: | ---: | ---: |"]
    for name in names:
        table.append(f"| {LABELS.get(name, name)} | {metrics['validation'][name]['rmse_mg_L']:.6g} | "
                     f"{metrics['test'][name]['rmse_mg_L']:.6g} | {metrics['future_time_test'][name]['rmse_mg_L']:.6g} |")
    noise_rmse = metrics["measurement_noise_sensitivity"]["scores"][selected]["rmse_mg_L"]
    improvement = 100*(1-test_score["rmse_mg_L"]/metrics["test"]["linear_drift"]["rmse_mg_L"])
    counts = metrics["counts"]
    sections = [
        ("Objective and completed scope", f"This project implements the B.Tech report's simulation-to-forecasting workflow for monoclonal antibody production. The completed computational deliverables are a corrected bioreactor simulator, synthetic multi-batch data generation, LSTM and alternative models, model selection, independent calibration and testing, saved inference, plots, a notebook, and this report. The result is a validated synthetic forecasting study, not a validated industrial monitoring system.\n\nThe best tested model for the declared clean synthetic task is {selected}: ridge regression with a cubic correction to the extrapolated local trend. Selection uses validation data only. This is a task-specific empirical choice, not a claim that ridge regression is universally better than LSTM or modern foundation models."),
        ("Mechanistic model and data provenance", f"The source is Kumar et al., Chemical Engineering and Processing 180 (2022), 108720, DOI 10.1016/j.cep.2021.108720. The simulator integrates volume, viable cells, mAb, glucose, lactate, and dead cells, deriving viability for reporting. The lactate equation uses the signed glucose rate and the printed grouping. Lacmax1 follows the reported 628 g/L; the original notebook used 6283.06289, whose fitting provenance remains unresolved.\n\nAll {sum(counts.values())} batches are numerical solutions with fixed kinetic parameters. Initial volume spans 4800-6500 mL, viable cells 0.45-0.85 million/mL, viability 97-100%, mAb 0-8 mg/L, glucose 6-9 g/L, and lactate 0.06-0.30 g/L. Temperature varies within 308-310 K, active flow within 0.25-1.5 mL/min, and feed glucose within 12-24 g/L. Four balanced modes are represented: batch, fed-batch, well-mixed continuous, and perfusion retaining cells and mAb. Controls stay constant during a run. There were {len(manifest['rejected_attempts'])} rejected simulation attempts. The dataset seed is {design['seed']}; all conditions and data hashes are saved in the manifest."),
        ("Forecast task and leakage controls", f"Input: {design['window']} observations of six reactor states and five operating settings, sampled every {design['step_min']} minutes. Twenty observations span 380 minutes; future targets are the next {design['horizon']} mAb values, covering 20-1000 minutes. No future glucose, cell counts, mAb, or other reactor states are supplied to the model. The supplied operating settings are assumed to remain constant.\n\nWhole batches are assigned to training ({counts['train']}), validation ({counts['validation']}), calibration ({counts['calibration']}), and final testing ({counts['test']}). Training and validation targets stop at 5000 minutes; development forecast origins run from 400 to 4000 minutes every 100 minutes. Each training batch contributes 37 windows, for 1776 training windows. No window crosses a batch boundary. Scaling is fitted on training histories only. Calibration is excluded from fitting and selection. Model selection is written to selection.json before test windows are constructed. A second test forecasts 5000-6000 minutes on the same unseen test batches, beyond every training target."),
        ("Models and training", "Persistence, linear drift, and local quadratic continuation are baselines. The learned models start from the last mAb value and last observed slope. They predict a smooth correction over the entire horizon: P_hat(h) = P_last + h*delta_P_last + r*(c1*u + c2*u^2 + c3*u^3), where h is the lead in samples, u=h/50, and r is a residual scale fitted on training data. The learned coefficients use only past measurements. This anchoring avoids relearning absolute concentration levels and avoids recursive feedback of predicted future inputs.\n\nRidge regression uses 33 features: the normalized current state/settings, the change across the history, and a midpoint curvature contrast. An intercept gives 34 regressors for three coefficients. Alpha is selected from 1e-6, 1e-4, 0.01, 1, and 100 using validation forecast RMSE. MLP uses the same feature summaries and two 64-unit SiLU layers. LSTM uses two 64-unit layers with 0.1 inter-layer dropout, plus a current-state skip connection and a small output head. Neural models use AdamW, learning rate 0.003, weight decay 1e-5, gradient clipping, validation-based scheduling, and early stopping. Three seeds (11, 29, 47) and their prediction ensembles are evaluated for each network. The original report's univariate two-layer LSTM remains available as a historical baseline in baselines/univariate_lstm.py; its old scores use a different task and must not be directly compared to this benchmark."),
        ("Results and interpretation", f"The selected model's validation RMSE is {metrics['validation'][selected]['rmse_mg_L']:.6g} mg/L. On {counts['test']} unseen batches, its pooled test RMSE is {test_score['rmse_mg_L']:.6g} mg/L and MAE is {test_score['mae_mg_L']:.6g} mg/L. This reduces error by {improvement:.3f}% relative to the linear-drift baseline for this clean synthetic task. The best validation-selected LSTM candidate is {best_lstm}, with test RMSE {metrics['test'][best_lstm]['rmse_mg_L']:.6g} mg/L.\n\nOn the later 5000-6000-minute test, the selected model's RMSE is {future_score['rmse_mg_L']:.6g} mg/L. The test RMSE at the 1000-minute lead is {test_score['by_lead'].get('1000', {}).get('rmse_mg_L', float('nan')):.6g} mg/L. The very small errors reflect a deterministic, smooth simulator with fixed kinetics and exact observations. They do not imply equal accuracy on laboratory data. Repeated windows in one batch are dependent; uncertainty in average batch performance is bootstrapped by batch, not by individual overlapping rows."),
        ("Uncertainty and sensitivity", f"A separate set of {counts['calibration']} calibration batches supplies one maximum absolute error score per batch, spanning all origins and forecast leads, including the later interval. The finite-sample 90% conformal rank gives a constant radius of {radius:.6g} mg/L. On the {counts['test']} clean synthetic test batches, observed simultaneous coverage is {metrics['intervals']['test_batch_coverage']:.1%}; point coverage is {metrics['intervals']['test_point_coverage']:.1%}. With only 12 calibration batches, the nominal 90% rule selects the largest calibration score and is conservative. This is a synthetic-distribution statement, not a clinical or experimental confidence claim.\n\nThe predeclared sensitivity test adds independent Gaussian perturbations with SD equal to 0.5% of each state's training standard deviation, leaving controls unchanged. The selected model's RMSE rises to {noise_rmse:.6g} mg/L. These perturbations are an illustrative stress test, not a measured sensor-noise distribution. Trend and curvature estimates amplify measurement errors. The supplied model and clean-data intervals should therefore not be represented as robust to noisy laboratory inputs. No model is retrained or selected using these test perturbations."),
        ("Project forecast and use", f"For the original project's 5000-minute history, the saved model forecasts the next 1000 minutes without solving the ODE. At 6000 minutes it predicts {project_score['final_forecast_mg_L']:.6g} mg/L. A separately generated reference continuation gives {project_score['final_simulator_reference_mg_L']:.6g} mg/L; forecast RMSE on this illustrative path is {project_score['rmse_mg_L']:.6g} mg/L. The reference is computed after selection and is not passed to inference.\n\nRun: python predict_mab.py --history data/example_batch.csv --operation configs/operation.example.json. The command loads best_model.pt and saves concentrations, interval bounds, and provenance metadata. It requires the original seven CSV columns, at least 20 compatible history timestamps, and the correct known constant controls. It does not interpolate missing timestamps or extrapolate beyond the trained 1000-minute horizon. It flags features substantially outside the empirical training range. For a shorter horizon use --minutes 240. To forecast from an earlier observed timestamp use --origin 4000; later CSV rows are ignored."),
        ("Reproduction, limitations, and next scientific step", "Install requirements.txt in a Python virtual environment. Run python run_project.py to regenerate all 88 batches, train candidates, select, calibrate, evaluate, forecast, and rebuild this report. Run python run_project.py --report-only to regenerate plots and the report from saved artifacts without retraining. Run python -m unittest discover -s tests -v for the regression and scientific-validity checks. requirements-lock.txt records the tested environment.\n\nThe completed deliverable is the reproducible computational B.Tech project. Independent experimental validation remains outside the available data. Before deployment, obtain measured runs, resolve the original kinetic fit and parameter discrepancy, model observation noise, recalibrate intervals, and evaluate whole held-out experimental batches. The present system assumes fixed kinetics, constant feed/temperature settings, and the declared operation modes. It does not model pH, implement plant control, optimize yield, or demonstrate antibody quality. Its strongest result is that a compact learned model can accurately emulate this simulator's forecast trajectories; a more complicated model is not automatically the best choice."),
    ]
    calibration_note = (" The usual distribution-free conformal guarantee requires exchangeable calibration "
                        "and future batches. This benchmark balances operating modes by design; pooled "
                        "exchangeability across modes has not been established. Interpret the bands as "
                        "nominal, empirically checked calibration, not a proven guarantee for each mode.")
    sections = [(title, body + calibration_note if title == "Uncertainty and sensitivity" else body)
                for title, body in sections]
    source_lines = [
        "project_proposal.pdf, supplied project brief, pages 3-11.",
        "Kumar et al. (2022), Multi-objective optimization of monoclonal antibody production in bioreactor. https://doi.org/10.1016/j.cep.2021.108720",
        "Hochreiter and Schmidhuber (1997), Long Short-Term Memory. Neural Computation 9(8), 1735-1780.",
        "Zeng et al., Are Transformers Effective for Time Series Forecasting? https://arxiv.org/abs/2205.13504. Motivation for testing simple models; the project does not claim to implement DLinear.",
        "Chen et al., TSMixer: An All-MLP Architecture for Time Series Forecasting. https://arxiv.org/abs/2303.06053. Motivation for a feed-forward comparison; the project MLP is not TSMixer.",
    ]
    markdown = ["# Final project report", "", "## Data-driven forecasting of mAb production", "",
                "Simulation-based comparison of LSTM, MLP, and regularized direct forecasting", ""]
    for title, body in sections:
        markdown += [f"## {title}", "", body, ""]
        if title == "Results and interpretation":
            markdown += ["All table entries are RMSE in mg/L; later-time test uses unseen batches at 5000-6000 minutes.", "", *table, "",
                         "![Model comparison](results/final/model_comparison.png)", "",
                         "![Held-out forecasts](results/final/heldout_forecasts.png)", ""]
        if title == "Uncertainty and sensitivity":
            markdown += ["![Error and noise sensitivity](results/final/error_and_noise.png)", ""]
        if title == "Project forecast and use":
            markdown += ["![Project forecast](results/final/project_forecast.png)", ""]
    markdown += ["## References", "", *[f"- {line}" for line in source_lines], ""]
    report_md = Path("docs/final_report.md") if out == Path("results/final") else out / "final_report.md"
    report_md.parent.mkdir(parents=True, exist_ok=True)
    image_prefix = os.path.relpath(out.resolve(), report_md.parent.resolve())
    report_md.write_text("\n".join(markdown).replace("](results/final/", f"]({image_prefix}/"))

    def text_page(title, body, number):
        page = plt.figure(figsize=(8.27, 11.69))
        page.text(.09, .94, "mAb FORECASTING | B.Tech PROJECT", fontsize=10, color="#16806B", weight="bold")
        page.text(.09, .90, title, fontsize=17, weight="bold", color="#25384B", va="top", wrap=True)
        lines = []
        for paragraph in body.split("\n\n"):
            lines.extend(textwrap.wrap(paragraph, width=88))
            lines.append("")
        page.text(.09, .84, "\n".join(lines), fontsize=10.5, va="top", linespacing=1.55)
        page.text(.09, .04, "Synthetic-data validation | Fixed kinetics | Constant controls", fontsize=8, color="gray")
        page.text(.92, .04, str(number), fontsize=9, ha="right")
        return page

    report_pdf = report_md.with_suffix(".pdf")
    with PdfPages(report_pdf) as pdf:
        for number, (title, body) in enumerate(sections, 1):
            page = text_page(title, body, number)
            pdf.savefig(page)
            plt.close(page)
            if title == "Results and interpretation":
                pdf.savefig(table_fig, bbox_inches="tight", pad_inches=.2)
                pdf.savefig(comparison_fig, bbox_inches="tight", pad_inches=.2)
                pdf.savefig(heldout_fig, bbox_inches="tight", pad_inches=.2)
            if title == "Uncertainty and sensitivity":
                pdf.savefig(sensitivity_fig, bbox_inches="tight", pad_inches=.2)
            if title == "Project forecast and use":
                pdf.savefig(project_fig, bbox_inches="tight", pad_inches=.2)
        page = text_page("References", "\n\n".join(source_lines), len(sections)+1)
        pdf.savefig(page)
        plt.close(page)
    for fig in (comparison_fig, table_fig, heldout_fig, sensitivity_fig, project_fig):
        plt.close(fig)
    print(f"Saved {report_md} and {report_pdf}")
    return report_md, report_pdf
