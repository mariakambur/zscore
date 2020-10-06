# Repository for z-score calculation app

To run z-score calculation application:
- make sure you have Python3.7 installed;
- run ```pip install -r requirements.txt``` to install proper versions of pandas and numpy;
- to calculate z-score and generate output file test_proc.tsv run application with options:

```python z_score.py /path/to/train.tsv /path/to/test.tsv normalization feature_code```

By default ```normalization```='z-score' and ```feature_code```='2' . 
