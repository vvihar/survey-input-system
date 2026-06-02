# survey_input_system

手書き済み調査票PDFから、調査票ID・数字記入式設問を自動で暫定読取し、Streamlit GUIで確認・訂正するための最小構成です。確認画面では調査票ページ全体の画像を保存し、設問ごとの表示範囲はブラウザ側で制御します。

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

# 3. 確認GUI用データセットを作る
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
- `review_work/pages/*.png`: 確認GUIで表示する調査票ページ画像
- `exports/*.csv`: 確定後CSV

## 確認画面の画像表示

`make_review_dataset` は設問ごとの切り出し画像を保存しません。冊子ページ単位の画像だけを `review_work/pages/` に保存し、各設問の `rect` / `context_rect` / `table_rect` を `review_items.jsonl` に記録します。Streamlit は現在の冊子の版と設問IDから `layout.json` の矩形を引き直し、CSS のクリップ表示でページ画像の該当範囲だけを表示します。

このため、Streamlit 上で冊子の版を訂正すると、同じページ画像に対して訂正版のレイアウト矩形が使われ、確認用の表示範囲も自動的に切り替わります。
