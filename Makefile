# One-line wrappers over the `uv run ...` commands specified per task in PLAN.md.
.PHONY: setup data validate-data eda features backtest train-final api dashboard drift-report test lint docker-up docker-down

setup:          ## Create/sync the virtualenv from uv.lock (P0.1)
	uv sync

data:           ## Download the Kaggle dataset into data/raw/ (P0.2)
	uv run python -m forecasting_platform.data.download

validate-data:  ## Validate data/raw/ against the expected schema (P0.2)
	uv run python -m forecasting_platform.data.validate

eda:            ## Execute the EDA notebook in place (P0.3)
	uv run jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb

features:       ## Build the feature frame into data/processed/ (P0.4)
	uv run python -m forecasting_platform.features.engineering

backtest:       ## Run every model tier through every CV fold (P0.7)
	uv run python scripts/run_backtest.py

train-final:    ## Retrain + register the champion model (P0.8)
	uv run python scripts/train_final_model.py

api:            ## Serve the FastAPI forecast service (P0.9)
	uv run uvicorn forecasting_platform.serving.api:app --reload

dashboard:      ## Launch the Streamlit dashboard (P0.10)
	uv run streamlit run dashboard/app.py

drift-report:   ## Generate the Evidently drift report (P0.11)
	uv run python scripts/generate_drift_report.py

test:           ## Run the test suite
	uv run pytest

lint:           ## Lint with ruff
	uv run ruff check .

docker-up:      ## Build and start api/dashboard/mlflow (P0.12)
	docker compose up --build

docker-down:    ## Stop the compose stack
	docker compose down
