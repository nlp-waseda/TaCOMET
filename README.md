# TaCOMET 

We built a time-aware commonsense knowledge model, TaCOMET.
Existing COMET models were trained to build TaCOMET, using event commonsense knowledge graphs with interval times between the events, TimeATOMIC.

This repository includes Japanese / English versions of TimeATOMIC and the links to the model pages in Hugging Face.
The training scripts are also included in each folder.

These were introduced in [NLP2024](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P3-19.pdf).

### Data
```
.
├── en
│   ├── data
│   │   ├── test.jsonl
│   │   └── train.jsonl
│   └── train.py
└── ja
    ├── data
    │   ├── mrph_test.jsonl
    │   ├── mrph_train.jsonl
    │   ├── test.jsonl
    │   └── train.jsonl
    └── train.py
```

```./*/data/``` contain TimeATOMIC split into train/test set.

```./ja/data/mrph_*.jsonl``` are ones segmented into words by [Juman++](https://github.com/ku-nlp/jumanpp).


### Models

The resulting TaCOMETs are available in Hugging Face.
Note that the Japanese one is trained with Juman++ pre-process.

| Language | TaCOMET | Base COMET |
| :------- | :------: | :-----: |
| Japanese | [link](https://huggingface.co/nlp-waseda/tacomet-gpt2-xl-japanese) | [link](https://huggingface.co/nlp-waseda/comet-gpt2-xl-japanese) |
| English  | [link](https://huggingface.co/nlp-waseda/tacomet-gpt2-xl-english) | [link](https://github.com/peterwestai2/symbolic-knowledge-distillation) $^{*}$ |

\* We used COMET-distil-high, introduced in [this paper](https://aclanthology.org/2022.naacl-main.341/), from this GitHub repo.

### References
```bibtex
@InProceedings{murata_nlp2023_tacomet,
    author =    "村田栄樹 and 河原大輔",
    title =     "TaCOMET: 時間を考慮したイベント常識生成モデル",
    booktitle = "言語処理学会第30回年次大会",
    year =      "2024",
    url =       "https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P3-19.pdf"
    note =      "in Japanese"
}

```
