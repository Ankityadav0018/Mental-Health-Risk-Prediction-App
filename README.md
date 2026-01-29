# Mental Health Risk Prediction App

This is a small Flask web app that uses pre-trained machine learning models (stored in the `models` folder) to predict mental health risk based on a form.

The project is already **macOS-safe** because it uses relative paths (no Windows-style paths or hard-coded directories). Follow the steps below to run it on your Mac.

## 1. Make sure Python is installed

On macOS, check:

```bash
python3 --version
```

If you don’t have Python 3, install it from the official site or using Homebrew:

```bash
brew install python
```

## 2. Create and activate a virtual environment (recommended)

From the project folder (`mlproject`):

```bash
cd /Users/ankityadav/Downloads/mlproject
python3 -m venv venv
source venv/bin/activate
```

If you later want to deactivate:

```bash
deactivate
```

## 3. Install required Python packages

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs:

- Flask
- NumPy
- joblib
- scikit-learn

## 4. Run the Flask app on macOS

From the project folder, with the virtual environment active:

```bash
python3 app.py
```

By default the app will run at:

```text
http://127.0.0.1:5000/
```

Open this URL in your browser to use the app.

## 5. Notes about models and templates

- The `models` folder already contains the trained artifacts:
  - `app_scaler.joblib`
  - `app_kmeans_model.joblib`
  - `app_cluster_risk_map.joblib`
- The `templates` folder contains:
  - `index.html` – input form
  - `result.html` – result page

As long as you keep this structure and run `python3 app.py` from the `mlproject` folder, everything will work correctly on your Mac.

