# TaCOMET 

We built a time-aware commonsense knowledge model, TaCOMET.
Existing COMET models were trained to build TaCOMET, using event commonsense knowledge graphs with interval times between the events, TimeATOMIC.

This repository includes Japanese / English versions of TimeATOMIC and the links to the model pages in Hugging Face.
The training scripts are also included in each folder.

These are introduced in [NLP2024](https://www.anlp.jp/proceedings/annual_meeting/2024/pdf_dir/P3-19.pdf) and [LREC-COLING2024](https://aclanthology.org/2024.lrec-main.1405/).

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
@inproceedings{murata-kawahara-2024-time-aware,
    title = "Time-aware {COMET}: A Commonsense Knowledge Model with Temporal Knowledge",
    author = "Murata, Eiki  and
      Kawahara, Daisuke",
    editor = "Calzolari, Nicoletta  and
      Kan, Min-Yen  and
      Hoste, Veronique  and
      Lenci, Alessandro  and
      Sakti, Sakriani  and
      Xue, Nianwen",
    booktitle = "Proceedings of the 2024 Joint International Conference on Computational Linguistics, Language Resources and Evaluation (LREC-COLING 2024)",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.lrec-main.1405",
    pages = "16162--16174",
    abstract = "To better handle commonsense knowledge, which is difficult to acquire in ordinary training of language models, commonsense knowledge graphs and commonsense knowledge models have been constructed. The former manually and symbolically represents commonsense, and the latter stores these graphs{'} knowledge in the models{'} parameters. However, the existing commonsense knowledge models that deal with events do not consider granularity or time axes. In this paper, we propose a time-aware commonsense knowledge model, TaCOMET. The construction of TaCOMET consists of two steps. First, we create TimeATOMIC using ChatGPT, which is a commonsense knowledge graph with time. Second, TaCOMET is built by continually finetuning an existing commonsense knowledge model on TimeATOMIC. TimeATOMIC and continual finetuning let the model make more time-aware generations with rich commonsense than the existing commonsense models. We also verify the applicability of TaCOMET on a robotic decision-making task. TaCOMET outperformed the existing commonsense knowledge model when proper times are input. Our dataset and models will be made publicly available.",
}
```
