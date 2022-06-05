# Answering Open-Domain Multi-Answer Questions via a Recall-then-Verify Framework
This repository includes the original implementation of our ACL2022 paper "[Answering Open-Domain Multi-Answer Questions via a Recall-then-Verify Framework][paper]", which contains the following two branches:
* retriever: code for finetuning a retriever
* rectify: code for training a recall-then-verify system

All data and checkpoints can be downloaded from [this url][data_url] (*comming soon*).

## Citation

If you find this work useful, please cite our paper:

```bibtex
@inproceedings{DBLP:conf/acl/ShaoH22,
  author    = {Zhihong Shao and
               Minlie Huang},
  editor    = {Smaranda Muresan and
               Preslav Nakov and
               Aline Villavicencio},
  title     = {Answering Open-Domain Multi-Answer Questions via a Recall-then-Verify
               Framework},
  booktitle = {Proceedings of the 60th Annual Meeting of the Association for Computational
               Linguistics (Volume 1: Long Papers), {ACL} 2022, Dublin, Ireland,
               May 22-27, 2022},
  pages     = {1825--1838},
  publisher = {Association for Computational Linguistics},
  year      = {2022},
  url       = {https://aclanthology.org/2022.acl-long.128},
  timestamp = {Wed, 18 May 2022 15:21:43 +0200},
  biburl    = {https://dblp.org/rec/conf/acl/ShaoH22.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

[paper]: https://arxiv.org/abs/2110.08544
[data_url]: https://comming_soon