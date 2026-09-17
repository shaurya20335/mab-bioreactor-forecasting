# Project review and corrections

The original notebook, CSV, Kumar et al. model equations and parameter tables,
and all 12 pages of the project brief were reviewed before correction.
See [references](references.md) and the preserved [project brief](project_proposal.pdf).
Downloaded papers, obsolete exports, and duplicate original implementation files
were removed during repository cleanup; reference links retain their provenance.

## Scientific transcription corrections

The reference is Kumar et al., *Chemical Engineering and Processing – Process
Intensification* 180 (2022), 108720,
[DOI: 10.1016/j.cep.2021.108720](https://doi.org/10.1016/j.cep.2021.108720).
Equations and the table were checked visually against the local PDF.

1. **Signed glucose rate in lactate production.** Equation (11), page 5,
   defines `q_glucose = -mu/yxs - ms`. Equation (12) uses
   `+ yls*q_glucose`. The notebook instead used `+ yls*(mu/yxs + ms)`,
   reversing that contribution. The implementation now follows the printed
   equation.
2. **Grouping in the lactate equation.** The factor
   `(1 - lactate/lacmax1)` applies to the entire sum
   `(mu/yxl + yls*q_glucose)` in equation (12). The notebook applied it only
   to the glucose-associated term. Both terms are now inside the multiplier.
3. **Parameter discrepancy.** Table 3, page 7, reports `Lacmax1 = 628 g/L`;
   the notebook used `6283.06289`. The default now uses the printed `628.0`.
   This is a choice to follow the publication, not proof that the original
   fitted parameter is wrong: the fitting code and underlying experimental
   data were not supplied. The remaining high-precision notebook parameters
   are retained and agree approximately with the table's rounded values.
4. **Source typo.** Equation (14) labels the mAb production rate `q_Glc`,
   despite its surrounding definition and equation (13) referring to mAb.
   The implementation names it `q_product`; it does not replace the glucose
   rate with the mAb production expression.

These changes align the implementation with the printed source. Experimental
validation is still required to establish predictive accuracy, particularly
because correcting the lactate equation changes the model used with the
original fitted parameter vector.

## Numerical and workflow corrections

- The original notebook integrated viability directly and reconstructed dead
  cells with `x*(100/viability - 1)`. This divided by zero at 0% viability and
  made the viability derivative undefined in an empty culture. Dead cells are
  now an explicit state; viability is derived only for reporting.
- The original denominator fallback added arbitrary small constants to two
  unrelated factors. Physical inputs are now validated, zero substrate/cells
  have a defined growth-rate limit, and the solver monitors domain boundaries.
- Extending the original batch to 15,000 minutes produced negative glucose
  and an `odeint` failure, with invalid output rows still available downstream.
  The new simulation checks solver status and stops with a descriptive error
  at depletion instead of exporting a failed trajectory. It also checks
  lactate depletion, the linear mAb inhibition boundary, and reactor emptying.
- Cells and mAb use the product rule for changing volume. The reference's
  retained perfusion outlet is represented separately from the well-mixed
  sample/harvest outlet. The original zero-flow batch balances for those
  species were already consistent; they were not changed arbitrarily.
- The notebook's 4,500-minute horizon conflicted with the CSV's 5,000-minute
  horizon. Both now use 5,000 minutes, with 5,001 samples. The original CSV
  matched the original equations over their shared time interval.
- Model parameters and operating conditions are named and configurable;
  equations are shared between the notebook and command-line script.
- Added dependency files, execution instructions, six labeled plots, explicit
  units, automated checks, and a standard Python notebook kernel.
- Regenerated CSV and notebook outputs describe the corrected implementation.
  The original project brief is preserved in `docs/project_proposal.pdf`.

## What is still outside the implementation

- The notebook uses a constant 308 K batch. The paper's experimental protocol
  includes feeding, sampling, perfusion, and a temperature change. Reactor run
  1's initial conditions alone do not make this a reproduction of that run.
- The supplied source had no ML training code despite the B.Tech report's
  LSTM results. `baselines/univariate_lstm.py` now supplies a reproducible implementation of
  that architecture, with new results described in `report_alignment.md`.
  PSO fitting and NSGA-II optimization are reference-paper methods, not
  requirements established by the B.Tech report, and are not implemented.
- The parameter table supplies rounded values; provenance of the notebook's
  additional digits and the `Lacmax1` discrepancy needs the original fit/data.
- Agreement with a generated CSV is a regression check, not experimental
  validation. No independent experimental dataset was supplied.
- Retention assumptions and the linear product inhibition expression belong
  to this particular model. They should be revisited before changing the
  process or extrapolating beyond the documented domain.

## Verification

The automated suite checks the signed lactate equation with a hand-computed
case; total-cell conservation during death; outlet/retention balances;
analytic mixing and withdrawal solutions; zero viable cells; physical default
outputs; convergence under tighter tolerances; depletion and inhibition
events; invalid inputs; solver failure propagation; and CSV column semantics.
It also checks chronological windows, exclusion of future data from scaling,
recursive prediction feedback, metric units, and checkpoint reloads. The final
extension adds whole-batch isolation, temporal holdouts, conformal batch
calibration, history resampling, and inference checks. The simulation notebook was executed
successfully in a Jupyter kernel; its CSV and PNG/PDF figures were regenerated.

At 5,000 minutes the corrected batch predicts about **161.624 mg/L mAb**
and **98.0773% viability**. These are simulation outputs, not measured yields.

The completed forecasting extension is documented in `final_report.md` and
`final_report.pdf`. It includes improved LSTM and MLP models and selects ridge
regression on validation batches. Clean synthetic test RMSE is 0.00521 mg/L;
the declared noise sensitivity test raises it to 6.80 mg/L. Prediction accuracy
and interval coverage must therefore be interpreted within the tested regime.
