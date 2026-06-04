# survey_input_system

手書き済み調査票PDFから、調査票ID・数字記入式設問を自動で暫定読取し、Streamlit GUIで確認・訂正するための最小構成です。確認画面では調査票ページ全体の画像を保存し、設問ごとの表示範囲はページ画像からメモリ上で一時的に切り出して表示します。

## 典型的な実行順序

```bash
uv sync

# スキャン済みPDFから暫定読取と確認GUI用データセット作成まで実行
uv run python -m main prepare -p scanned/scanned-sheet.pdf

# ディレクトリ単位で処理する場合
uv run python -m main prepare -d scanned/

# GUIで確認・訂正
uv run streamlit run app_streamlit.py -- --workdir <prepareで表示されたreviewディレクトリ>

# CSV出力
uv run python -m scripts.export_csv --workdir <reviewディレクトリ> --out-dir <reviewディレクトリ>/exports
```

`prepare` は、既定で `survey-sheet.pdf`、`survey-sheet.tex`、`models/mnist.pt` を使います。レイアウトJSONがなければ `outputs/layout_survey.json` を自動生成し、読取結果を `outputs/answers/`、確認GUI用データを `outputs/review/` または `outputs/review_merged/` に保存します。別のテンプレートや出力先を使う場合だけ `--template-pdf`、`--tex`、`--model`、`--out-dir` を指定してください。

## 想定する入力

- `template-pdf`: A/B/C 版を各4ページずつ含む空欄テンプレートPDF
- `answered-pdf`: スキャン済み回答PDF。4ページで1冊子。複数冊子連結PDFでも可
- `tex`: 調査票のTeXソース
- `model`: 学習済みMNIST数字分類モデル。不要なら `--model` は省略可

## 出力

- `outputs/layout_survey.json`: テンプレートPDF上の設問座標とTeX由来の設問メタデータ
- `outputs/answers/*.json`: 調査票ID、版、数字記入式設問の暫定読取結果
- `outputs/review/*/review_items.jsonl`: GUI用の設問単位データ
- `outputs/review/*/pages/*.png`: 確認GUIで表示する調査票ページ画像
- `outputs/review/*/exports/*.csv`: 確定後CSV

複数PDFを扱う場合、PDF由来の中間生成物名には `PDF名_パス由来ハッシュ` のキーを使います。同じファイル名のPDFが別ディレクトリにあっても、`review_batch` のサブディレクトリ、ページ画像、Streamlit widget IDが衝突しないようにしています。

## 確認画面の画像表示

`make_review_dataset` は設問ごとの切り出し画像を保存しません。冊子ページ単位の画像だけを `outputs/review/.../pages/` に保存し、各設問の `rect` / `context_rect` / `table_rect` を `review_items.jsonl` に記録します。Streamlit は現在の冊子の版と設問IDから `layout.json` の矩形を引き直し、ページ画像から該当範囲をメモリ上で一時的に切り出して表示します。

このため、Streamlit 上で冊子の版を訂正すると、同じページ画像に対して訂正版のレイアウト矩形が使われ、確認用の表示範囲も自動的に切り替わります。

通常の設問確認では context 画像内の回答欄をハイライト表示します。回答欄だけを拡大確認したい場合は、各設問の「回答欄を単独表示」を有効にしてください。冊子メタ情報の該当部分画像も、上部パネル内の「該当部分の画像を表示」を有効にした場合だけ読み込みます。

複数回答の数字設問は `1,3,5` のようにカンマ区切りで入力できます。旧形式や予測値由来の連結数字は、保存時にカンマ区切りへ正規化されます。
