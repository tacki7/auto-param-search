# auto-param-search

Search for arithmetic combinations of columns in a tabular dataset (CSV / Parquet)
that correlate strongly with a target column.

- `ga_feature_search.py` — genetic algorithm over expression trees (`+ - * /`), pure NumPy/pandas.
- `ga_pysr_search.py` — same idea backed by [PySR](https://github.com/MilesCranmer/PySR) symbolic regression.

Both print the top expressions with their correlation and save a math-rendered image.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python ga_feature_search.py mpg.csv --target mpg
python ga_pysr_search.py mpg.csv --target mpg --time-limit 60
```

Run with `--help` for all options.

## Sample data

- `mpg.csv` — Auto MPG dataset
- `sample.csv` — synthetic data
- `mpg_results.png`, `mpg_math.png` — example outputs

## License

[MIT](LICENSE)
