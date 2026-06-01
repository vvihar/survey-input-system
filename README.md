# survey_input_system

手書き済み調査票PDFから、調査票ID・数字記入式設問を自動で暫定読取し、数字欄・5段階評価欄・活動表を画像切り出しして、Streamlit GUIで確認・訂正するための最小構成です。

## 典型的な実行順序

```bash
pip install -r requirements.txt

# 1. テンプレートPDFとTeXからレイアウトJSONを作る
python -m scripts.build_layout \
  --template-pdf survey-sheet.pdf \
  --tex survey-sheet.tex \
  --out layout_survey.json

# 2. スキャン済みPDFから調査票ID・数字記入欄を暫定読取してJSONへ
python -m scripts.read_scan \
  --answered-pdf scanned/scanned1.pdf \
  --template-pdf survey-sheet.pdf \
  --layout layout_survey.json \
  --model models/mnist.pt \
  --out answers_initial.json

# 3. 各設問を画像として切り出し、GUI用データセットを作る
python -m scripts.make_review_dataset \
  --answered-pdf scanned/scanned1.pdf \
  --template-pdf survey-sheet.pdf \
  --layout layout_survey.json \
  --answers answers_initial.json \
  --workdir review_work

# 複数PDF/ディレクトリ単位の一括作成
python -m scripts.make_review_dataset_batch \
  --answered-dir scanned/ \
  --template-pdf survey-sheet.pdf \
  --layout layout_survey.json \
  --answers-dir answers/ \
  --workdir review_work_batch

# 4. GUIで確認・訂正
streamlit run app_streamlit.py -- --workdir review_work

# 5. CSV出力
python -m scripts.export_csv --workdir review_work --out-dir review_work/exports
```

## 想定する入力

- `template-pdf`: A/B/C 版を各4ページずつ含む空欄テンプレートPDF
- `answered-pdf`: スキャン済み回答PDF。4ページで1冊子。複数冊子連結PDFでも可
- `tex`: 調査票のTeXソース
- `model`: 学習済みMNIST風数字分類モデル。不要なら `--model` は省略可

## 出力

- `layout_survey.json`: テンプレートPDF上の設問座標とTeX由来の設問メタデータ
- `answers_initial.json`: 調査票ID、版、数字記入式設問の暫定読取結果
- `review_work/review_items.jsonl`: GUI用の設問単位データ
- `review_work/crops/*.png`: 切り出し画像
- `exports/*.csv`: 確定後CSV
