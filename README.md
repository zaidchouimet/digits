# Digit Recognition Benchmark System

Scientifically valid benchmarking for printed digit recognition. The project separates detection from recognition, supports OCR-only baselines, records system metrics, and keeps PaddleOCR optional so the project still runs on Python 3.14 with EasyOCR fallback.

## Project structure

```text
digits/
├── benchmark/
│   ├── data_sources.py
│   ├── runner.py
│   └── storage.py
├── detection/
│   ├── base.py
│   ├── efficientdet.py
│   ├── full_frame.py
│   └── yolo.py
├── recognition/
│   ├── base.py
│   ├── easyocr_recognizer.py
│   ├── paddleocr_recognizer.py
│   └── tesseract_recognizer.py
├── pipelines/
│   ├── base_pipeline.py
│   ├── efficientdet_paddleocr.py
│   ├── ocr_only.py
│   ├── paddleocr_full.py
│   ├── yolo_easyocr.py
│   ├── yolo_paddleocr.py
│   └── yolo_tesseract.py
├── scripts/
│   └── setup_py310_env.ps1
├── utils/
├── main.py
├── requirements.txt
└── requirements-py310.txt
```

## Valid benchmark design

The default benchmark path is:

```text
image or frame -> detector -> crop regions -> OCR recognizer -> digit sequence
```

Available academic comparison groups:

- `Detection + OCR`: YOLO or EfficientDet detector paired with Tesseract, EasyOCR, or PaddleOCR
- `OCR-Only Baseline`: full-image OCR without a separate detector

This avoids the incorrect comparison of "YOLO versus OCR" as if they solved the same task.

## Environment strategy

The current workspace uses Python 3.14. PaddleOCR does not install there on Windows, so the project is designed to behave like this:

- On Python 3.14: EasyOCR and Tesseract pipelines work. PaddleOCR pipelines load and automatically fall back to EasyOCR.
- On Python 3.10 to 3.13: PaddleOCR can be installed normally and used directly.

### Recommended Windows setup for PaddleOCR

Install Python 3.10, then run:

```powershell
.\scripts\setup_py310_env.ps1
```

That script creates `venv310` and installs `requirements-py310.txt`.

Manual equivalent:

```powershell
py -3.10 -m venv venv310
venv310\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-py310.txt
```

## Running the benchmark

### CLI

List pipelines:

```powershell
python main.py --list-pipelines
```

Run a benchmark:

```powershell
python main.py --pipeline yolo_paddleocr
python main.py --pipeline yolo26n_easyocr --data-source svhn --svhn-size 100
python main.py --pipeline ocr_only --data-source images --image-dir .\images
```

Results are appended to `results.csv`.

### Streamlit dashboard

```powershell
streamlit run main.py
```

## Benchmark metrics

The benchmark records:

- throughput: FPS
- latency: average per-sample latency in milliseconds
- system load: CPU usage, memory usage, peak memory, energy
- recognition quality: CER, WER, digit accuracy, sequence accuracy

## PaddleOCR fallback behavior

Paddle-based recognizers try PaddleOCR first. If that import or load fails, they log a warning and switch to EasyOCR automatically. This keeps the project usable in unsupported environments without hiding the active backend from the benchmark report.
