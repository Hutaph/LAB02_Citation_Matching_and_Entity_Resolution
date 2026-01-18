# LAB02 - Citation Matching & Entity Resolution

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Active-success.svg)

> Hệ thống tự động khớp citation trong tài liệu khoa học sử dụng Machine Learning để giải quyết bài toán Entity Resolution trong scientific papers.

---

## Mục lục

- [Giới thiệu](#giới-thiệu)
- [Thông tin sinh viên](#thông-tin-sinh-viên)
- [Kiến trúc hệ thống](#kiến-trúc-hệ-thống)
- [Cài đặt](#cài-đặt)
- [Hướng dẫn sử dụng](#hướng-dẫn-sử-dụng)
  - [Bước 1: Parse LaTeX Files](#bước-1-parse-latex-files)
  - [Bước 2: Matching & Feature Extraction](#bước-2-matching--feature-extraction)
  - [Bước 3: Training & Evaluation](#bước-3-training--evaluation)
- [Cấu trúc dữ liệu](#cấu-trúc-dữ-liệu)
- [Kết quả](#kết-quả)
- [Cấu trúc thư mục](#cấu-trúc-thư-mục)

---

## Giới thiệu

Dự án này giải quyết bài toán **Citation Matching** - một dạng của **Entity Resolution** trong lĩnh vực xử lý tài liệu khoa học. Hệ thống tự động:

- **Trích xuất** citation references từ LaTeX source code
- **Khớp** các citation keys với các papers trong database
- **Huấn luyện** mô hình Machine Learning (Random Forest) để dự đoán matching
- **Đánh giá** bằng chỉ số MRR@5 (Mean Reciprocal Rank)

### Tính năng chính

- Parse LaTeX files và trích xuất bibliography items
- Matching thông minh giữa citation keys và reference papers
- Feature engineering với 11 features dùng cho huấn luyện
- Train/test split theo publication-level
- Hyperparameter tuning với GridSearchCV
- Evaluation với MRR@5 metric

### Công nghệ sử dụng

- **Python 3.10+** - Ngôn ngữ lập trình chính
- **Jupyter Notebook** - Môi trường phát triển
- **scikit-learn** - Machine Learning framework
- **pandas** - Xử lý dữ liệu
- **numpy** - Tính toán số học
- **matplotlib, seaborn** - Visualization
- **joblib** - Model serialization
- **tqdm** - Progress bars

---

## Thông tin sinh viên

| Thông tin | Giá trị |
|-----------|---------|
| **MSSV** | 23120334 |
| **Họ và tên** | Huỳnh Tấn Phước |
| **Email** | 23120334@student.hcmus.edu.vn |
| **Môn học** | Nhập môn Khoa học dữ liệu |
| **Lớp** | CQ2023/21 |

---

## Kiến trúc hệ thống

Hệ thống được chia thành 3 giai đoạn chính:

```
┌─────────────────────────────────────────────────────────────┐
│                    WORKFLOW PIPELINE                         │
└─────────────────────────────────────────────────────────────┘

1. PARSE STAGE
   └─> LaTeX files → bibitems.jsonl + references.jsonl

2. MATCH & FE STAGE  
   └─> bibitems + references → matches_fe.jsonl
   └─> Split train/val/test
   └─> Generate pred.json

3. MODELING STAGE
   └─> Train Random Forest
   └─> Hyperparameter Tuning
   └─> Evaluate MRR@5
```

### Pipeline chi tiết

1. **Parse LaTeX Files** (`parse_runner.ipynb`)
   - Đọc LaTeX source từ thư mục `23120334/{paper_id}/tex/`
   - Trích xuất bibliography items từ `.bib` files
   - Trích xuất references từ `.tex` files
   - Tạo `aggregated/bibitems.jsonl` và `aggregated/references.jsonl`

2. **Matching & Feature Extraction** (`match_and_fe.ipynb`)
   - Khớp citation keys với reference papers
   - Tính toán features dùng cho mô hình

#### Model Features

| Feature | Công thức/Tính toán | Sử dụng | Lý do |
|---------|-------------------|---------|-------|
| **levenshtein** | `1 - (edit_distance / max(len(a), len(b)))` | Có | Đo độ tương đồng chuỗi, quan trọng cho text matching |
| **year_match** | `1` nếu `source_year == cand_year`, `0` nếu không | Có | Feature quan trọng (importance ~0.049) |
| **year_diff** | `\|source_year - cand_year\|` hoặc `100` nếu thiếu | Có | Feature quan trọng thứ 2 (importance ~0.213) |
| **source_year** | Năm xuất bản từ bibitem | Có | Context feature cho model |
| **cand_year** | Năm xuất bản từ reference | Có | Context feature cho model |
| **author_overlap** | Jaccard overlap tên đầy đủ: `\|set(authors_a) & set(authors_b)\| / \|set(authors_a) \| set(authors_b)\|` | Có | Feature quan trọng (importance ~0.039) |
| **author_lastname_match** | `1` nếu có bất kỳ tên họ trùng, `0` nếu không | Có | Feature quan trọng nhất (importance ~0.587) |
| **author_firstname_match** | `1` nếu có bất kỳ tên đầu trùng, `0` nếu không | Không | Bị loại bỏ - không được tính trong compute_features |
| **token_overlap** | `\|set(tokens_a) & set(tokens_b)\|` | Không | Bị loại bỏ - trùng với author_overlap, char n-grams |
| **token_overlap_ratio** | `\|set(tokens_a) & set(tokens_b)\| / max(len(tokens_a), len(tokens_b))` | Không | Bị loại bỏ - trùng với token_overlap |
| **char_ngram_3** | Jaccard overlap 3-gram ký tự | Không | Bị loại bỏ - redundant với levenshtein |
| **char_ngram_4** | Jaccard overlap 4-gram ký tự | Không | Bị loại bỏ - redundant với levenshtein |
| **char_ngram_5** | Jaccard overlap 5-gram ký tự | Không | Bị loại bỏ - redundant với levenshtein |

   - Negative sampling (5000 negatives per positive)
   - Tạo train/val/test split (publication-level)
   - Sinh `pred.json` cho mỗi paper

3. **Modeling & Evaluation** (`modeling.ipynb`)
   - Load dữ liệu từ `split/` directory
   - Hyperparameter tuning với GridSearchCV
   - Train Random Forest Classifier
   - Đánh giá bằng MRR@5 metric
   - Tạo predictions và cập nhật `pred.json`

---

## Cài đặt

### Yêu cầu hệ thống

- Python 3.10 trở lên
- pip package manager
- ~10GB dung lượng ổ cứng (cho dataset và models)

### Thiết lập môi trường

#### Windows

```bash
# Tạo môi trường ảo
python -m venv .venv

# Kích hoạt môi trường
.venv\Scripts\activate

# Nâng cấp pip
pip install --upgrade pip

# Cài đặt dependencies
pip install -r requirements.txt
```

#### Linux/macOS

```bash
# Tạo môi trường ảo
python3 -m venv .venv

# Kích hoạt môi trường
source .venv/bin/activate

# Nâng cấp pip
pip install --upgrade pip

# Cài đặt dependencies
pip install -r src/requirements.txt
```

#### Google Colab (Khuyến nghị)

**📁 Dữ liệu dự án**: [Google Drive](https://drive.google.com/drive/folders/1RJC81xq4osFdIOGtwy_pKQoxlwGW3FZC?usp=sharing)

**Cấu trúc thư mục trong Drive**:
```
23120334/
├── aggregated/      # Dữ liệu đã aggregate
├── notebooks/       # Jupyter notebooks
├── src/            # Source code
└── 23120334/       # Papers data
```

**Thiết lập trên Colab**:

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Chuyển đến thư mục dự án
%cd "/content/drive/MyDrive/23120334"

# Cài đặt dependencies
!pip install -r requirements.txt
```

**Lưu ý**: Dự án được thiết kế để chạy trên Google Colab. Vui lòng tải toàn bộ thư mục từ Google Drive và mount vào Colab theo cấu trúc trên.

### Dependencies chính

- `pandas` - Xử lý dữ liệu
- `scikit-learn` - Machine Learning
- `numpy` - Tính toán số học
- `matplotlib`, `seaborn` - Visualization
- `tqdm` - Progress bars
- `joblib` - Model serialization

---

## Hướng dẫn sử dụng

> **⚠️ Lưu ý**: Dự án được thiết kế để chạy trên **Google Colab**. Vui lòng tải dữ liệu từ [Google Drive](https://drive.google.com/drive/folders/1RJC81xq4osFdIOGtwy_pKQoxlwGW3FZC?usp=sharing) và mount vào Colab theo cấu trúc thư mục đã cung cấp.

### Bước 1: Parse LaTeX Files

Notebook: `notebooks/parse_runner.ipynb`

**Mục đích**: Trích xuất bibliography items và references từ LaTeX source code.

**Cấu hình**:

```python
RUN_ALL = True           # True: xử lý nhiều paper; False: chỉ 1 paper
PAPER_ID = "2408-02468"  # Dùng khi RUN_ALL=False
START = "2408-02468"     # Dùng khi RUN_ALL=True
NUM = 5000               # Giới hạn số paper
```

**Chạy (theo thứ tự cell)**:

1. Cell cấu hình + import (đặt `RUN_ALL`, `PAPER_ID`, `START`, `NUM`)
2. Cell "Run Parser"
3. Cell "Statistics"
4. Cell "Visualization" (tùy chọn)
5. Cell "Quick Check"

**Kết quả**:
   - `aggregated/bibitems.jsonl`
   - `aggregated/references.jsonl`

**Output mẫu** (`bibitems.jsonl`):

```json
{
  "paper_id": "2408-02468",
  "key": "zhang2021counterfactual",
  "title": "Counterfactual Learning...",
  "authors": ["Zhang", "Li"],
  "year": 2021,
  "arxiv": "2112.12938"
}
```

---

### Bước 2: Matching & Feature Extraction

Notebook: `notebooks/match_and_fe.ipynb`

**Mục đích**: 
- Khớp citation keys với reference papers
- Tính toán features
- Tạo train/val/test split
- Sinh file `pred.json` cho mỗi paper

**Cấu hình**:

```python
NEG_PER_POS = 5000       # Số negative samples per positive
RANDOM_SEED = 23120334   # Random seed để reproducibility
START = "2408-02468"     # Lọc paper từ ID này
NUM = 600                # Số paper tối đa
MAX_REFS = None          # Giới hạn references (None = không giới hạn)
MAX_BIBS = None          # Giới hạn bibitems (None = không giới hạn)
```

**Chạy (theo thứ tự cell)**:

1. Cell cấu hình + import (đặt `NEG_PER_POS`, `START`, `NUM`, `MAX_REFS`, `MAX_BIBS`)
2. Cell "Load Manual Candidates" (nếu có)
3. Cell "Run Matching and Feature Extraction"
4. Cell "Statistics"
5. Cell "Visualization" (tùy chọn)
6. Cell "Data Splitting" để tạo `split/` và `pred.json`

**Output**:

- `aggregated/matches_fe.jsonl`: Tất cả cặp (bib_key, candidate) với features
- `split/train.jsonl`: Training set
- `split/val.jsonl`: Validation set  
- `split/test.jsonl`: Test set
- `23120334/{paper_id}/pred.json`: Predictions cho mỗi paper

**Thống kê split**:

```
Split sizes (papers): {'test': 2, 'train': 528, 'val': 2}
partition
train    489406
test       2597
val        2574
```

---

### Bước 3: Training & Evaluation

Notebook: `notebooks/modeling.ipynb`

**Mục đích**:
- Train Random Forest model
- Hyperparameter tuning
- Đánh giá bằng MRR@5

**Cấu hình Optuna Hyperparameter Tuning**:

```python
# Optuna search space
N_TRIALS = 30
RANDOM_STATE = 42

params = {
    'n_estimators': trial.suggest_int('n_estimators', 100, 400, step=50),
    'max_depth': trial.suggest_int('max_depth', 5, 30, step=5),
    'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5),
    'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
    'class_weight': trial.suggest_categorical('class_weight', ['balanced', 'balanced_subsample']),
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}
```

**Chạy (theo thứ tự cell)**:

1. Cell cấu hình + "Load Data"
2. Cell "Data Split Summary"
3. Cell "Feature Configuration"
4. Cell "Model Training" (Optuna)
5. Cell "Model Evaluation"
6. Cell "Feature Importance Analysis" (tùy chọn)
7. Cell "Generate Predictions"

**Kết quả**:

```
Best Parameters (Optuna):
{
    'n_estimators': 350,
    'max_depth': 15,
    'min_samples_split': 2,
    'min_samples_leaf': 5,
    'max_features': 'log2',
    'class_weight': 'balanced_subsample'
}

Best Val F1: 0.7072
Test MRR@5: 0.8911
Result: Excellent (>0.7)
```

**Model được lưu tại**: `models/citation_matcher_rf.pkl`

---

## Cấu trúc dữ liệu

### bibitems.jsonl

Mỗi dòng là một bibliography item:

```json
{
  "paper_id": "2408-02468",
  "key": "zhang2021counterfactual",
  "title": "Counterfactual Learning for Recommendation",
  "authors": ["Zhang", "Li", "Wang"],
  "year": 2021,
  "arxiv": "2112.12938",
  "venue": "arXiv"
}
```

### references.jsonl

Mỗi dòng là một reference paper:

```json
{
  "paper_id": "2408-02468",
  "id": "2112.12938",
  "arxiv": "2112.12938",
  "title": "Counterfactual Learning for Recommendation",
  "authors": ["Zhang", "Li", "Wang"],
  "year": 2021
}
```

### matches_fe.jsonl

Mỗi dòng là một cặp (bib_key, candidate) với features:

```json
{
  "paper_id": "2408-02468",
  "bib_key": "zhang2021counterfactual",
  "cand_id": "2112.12938",
  "score": 1.0,
  "levenshtein": 0.95,
  "jaccard": 0.88,
  "year_match": 1,
  "year_diff": 0,
  "source_year": 2021,
  "cand_year": 2021,
  "label": 1
}
```

### pred.json

File dự đoán cho mỗi paper:

```json
{
  "partition": "test",
  "groundtruth": {
    "zhang2021counterfactual": "2112.12938",
    "li2022deep": "2201.12345"
  },
  "prediction": {
    "zhang2021counterfactual": ["2112.12938", "2112.12939", "2112.12940", "2112.12941", "2112.12942"],
    "li2022deep": ["2201.12345", "2201.12346", "2201.12347", "2201.12348", "2201.12349"]
  }
}
```

### manual_candidates.json

File chứa các mapping thủ công (manual labels) từ bib_key sang arxiv_id cho các papers. Được sử dụng để tạo ground truth labels trong quá trình training:

```json
{
  "2408-02468": {
    "zhang2021counterfactual": "2112.12938",
    "li2022deep": "2201.12345"
  },
  "2408-02469": {
    "wang2023neural": "2301.12345",
    "chen2022transformer": "2205.67890"
  }
}
```

**Cấu trúc**: Mỗi key là `paper_id`, giá trị là một dictionary mapping `bib_key` → `arxiv_id` (candidate ID).

---

## Kết quả

### Performance Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **MRR@5** | 0.8911 | Excellent (>0.7) |
| **Test Queries** | 114 | Number of test citation keys |
| **Precision** | 0.98 | For class 1 (match) |
| **Recall** | 0.87 | For class 1 (match) |
| **F1-Score** | 0.92 | For class 1 (match) |

### Feature Importance

Theo thứ tự quan trọng (với excluded features):

1. **author_lastname_match** - 0.587 (Feature quan trọng nhất)
2. **year_diff** - 0.213 (Chênh lệch năm)
3. **year_match** - 0.049 (Khớp năm)
4. **cand_year** - 0.046 (Năm candidate)
5. **source_year** - 0.046 (Năm source)
6. **author_overlap** - 0.039 (Overlap tác giả)
7. **levenshtein** - 0.020 (Độ tương đồng chuỗi)

### Model Configuration

```python
RandomForestClassifier(
    n_estimators=350,
    max_depth=15,
    min_samples_split=2,
    min_samples_leaf=5,
    max_features='log2',
    class_weight='balanced_subsample',
    random_state=42,
    n_jobs=-1
)
```

---

## Cấu trúc thư mục

```
23120334/                          # Thư mục gốc dự án
├── src/                           # Source code chính
│   ├── parse_util.py             # Parser LaTeX
│   ├── match_utils.py            # Matching pipeline (tích hợp TF-IDF matching và text normalization)
│   ├── feature_engineering_utils.py  # Feature computation
│   ├── modeling_utils.py         # Training & evaluation
│   └── visualization.py          # Visualization helpers
│
├── notebooks/                     # Jupyter notebooks
│   ├── parse_runner.ipynb        # Bước 1: Parse LaTeX
│   ├── match_and_fe.ipynb        # Bước 2: Matching & FE
│   └── modeling.ipynb            # Bước 3: Training & Evaluation
│
├── aggregated/                    # Dữ liệu đã aggregate
│   ├── bibitems.jsonl            # Bibliography items
│   ├── references.jsonl          # References
│   ├── matches_fe.jsonl          # Matched pairs với features
│   └── manual_candidates.json    # Manual labels (nếu có)
│
├── split/                         # Train/Val/Test splits
│   ├── train.jsonl               # Training set
│   ├── val.jsonl                 # Validation set
│   └── test.jsonl                # Test set
│
├── models/                        # Trained models
│   └── citation_matcher_rf.pkl   # Random Forest model
│
├── 23120334/                      # Papers data
│   ├── 2408-02468/
│   │   ├── metadata.json
│   │   ├── references.json
│   │   ├── refs.bib               # Bibliography (generated)
│   │   ├── hierarchy.json         # Citation hierarchy (generated)
│   │   ├── pred.json             # Predictions
│   └── ...
├── requirements.txt              # Dependencies
└── README.md                     
```

### Module Dependencies

**`match_utils.py`** - Module trung tâm cho matching pipeline:
- **Tích hợp đầy đủ**: Chứa tất cả các hàm cần thiết cho TF-IDF matching và text normalization
- **Text normalization**: Bao gồm `normalize_ref_text()`, `cleanup_formatting()`, `normalize_spaces()`, `protect_math()` (tích hợp từ `latex_parser_tree.py`)
- **TF-IDF matching**: Bao gồm `build_text()`, `compute_matches()` (tích hợp từ `reference_matcher_tfidf.py`)
- **Không có external dependencies**: Không import từ `latex_parser_tree` hay `reference_matcher_tfidf` để tránh circular imports

**`feature_engineering_utils.py`**:
- Cung cấp các hàm tính toán features: `compute_features()`, `author_overlap()`, `author_lastname_match()`, `parse_year_int()`
- Được sử dụng bởi `match_utils.py` và `modeling_utils.py`

**Các modules khác**:
- `parse_util.py`: Parser LaTeX độc lập
- `modeling_utils.py`: Training và evaluation, sử dụng `feature_engineering_utils.py`
- `visualization.py`: Visualization helpers độc lập

---

## Troubleshooting

### Lỗi: MemoryError khi training

**Giải pháp**:
- Giảm `NUM` trong `match_and_fe.ipynb` (từ 600 xuống 300)
- Giảm `n_jobs` trong GridSearchCV (từ 2 xuống 1)
- Sử dụng subsample nhỏ hơn cho negative sampling

### Lỗi: Missing split files

**Giải pháp**:
- Đảm bảo đã chạy `match_and_fe.ipynb` đến cell cuối (split & pred.json)
- Kiểm tra `split/` directory có đầy đủ 3 files

### Lỗi: ImportError khi chạy notebooks

**Giải pháp**:
- Đảm bảo đã cài đặt dependencies: `pip install -r requirements.txt`
- Kiểm tra `sys.path` trong notebook có trỏ đúng đến `src/`
- Lưu ý: `match_utils.py` đã tích hợp đầy đủ các functions cần thiết, không cần import từ `latex_parser_tree` hay `reference_matcher_tfidf`

---

## Tài liệu tham khảo

- [scikit-learn Random Forest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Mean Reciprocal Rank (MRR)](https://en.wikipedia.org/wiki/Mean_reciprocal_rank)
- [Entity Resolution Overview](https://en.wikipedia.org/wiki/Record_linkage)

---

## Acknowledgments

- Dataset từ arXiv
- Semantic Scholar API cho reference data
- scikit-learn team cho ML framework

---

**Liên hệ**: 23120334@student.hcmus.edu.vn 

---

