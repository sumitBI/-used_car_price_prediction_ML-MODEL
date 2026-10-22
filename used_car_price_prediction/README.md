# Used Car Price Prediction — CarDekho

A reproducible supervised regression project for **historical** used-car prices in INR.

## Data
Original dataset: `data/raw/car_details.csv` (4,340 records, eight columns). The data has repeated identical records, so we group identical feature combinations in both train/test and cross-validation splits. No verified vehicle IDs or sale dates are available.

## Reproduce
1. Create an environment: `python -m venv .venv` and activate it.
2. Install dependencies: `pip install -r requirements.txt`.
3. Open Jupyter from the project root (`jupyter notebook`).
4. Run `notebooks/06_model_training.ipynb`, then `07_model_evaluation.ipynb`, then `08_hyperparameter_tuning.ipynb` in order. Each notebook detects the project root when launched from the root or `notebooks` folder.
5. Run the app: `streamlit run app/streamlit_app.py`.

## Methodology
- Features: year, km driven, fuel, seller type, transmission, ownership.
- Training-only grouped 5-fold CV chooses a candidate by MAE.
- Validation notebook analyzes out-of-fold residuals, without opening the test set.
- Tuning notebook tunes Random Forest, selects by training-only CV, and evaluates once on held-out data.
- Preprocessing lives inside the model pipeline to avoid leakage.

## Generated files
- `reports/cv_model_comparison.csv`: CV scores (run notebook 06)
- `reports/validation_error_by_price_segment.csv`: validation errors (run notebook 07)
- `reports/final_test_metrics.json`: final held-out metrics (run notebook 08)
- `models/best_car_price_model.joblib`: final trained pipeline (run notebook 08)

## Limitations
Historical listings are not 2026 market values. Dataset lacks car condition, location, vehicle IDs and verified sale dates. Do not use estimates as guaranteed offers. Do not load untrusted `.joblib` files.
