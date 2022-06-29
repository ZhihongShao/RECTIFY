# Answering Open-Domain Multi-Answer Questions via a Recall-then-Verify Framework
This repository includes the original implementation of our ACL2022 paper "[Answering Open-Domain Multi-Answer Questions via a Recall-then-Verify Framework][paper]", which contains the following two branches:
* retriever: code for finetuning a retriever
* rectify: code for training a recall-then-verify system

Download links for data and checkpoints:
* All source data and inferred results: [https://cloud.tsinghua.edu.cn/d/491c750515b94c73bbbb/][data_url]
* Finetuned retrievers and inferred passage embeddings: [https://cloud.tsinghua.edu.cn/d/85c358be556e4ce7b8ec/][retriever_url]
* Recallers and verifiers: [https://cloud.tsinghua.edu.cn/d/71a2177eefb34a33848c/][rectify_url]

***Due to the limited upload size, zip files have been chunked into smaller ones whose names share the same prefix except the last letter (e.g., `data.zipchunkaa` and `data.zipchunkab` are the first and the second chunk of the original zip file `data.zip`, respectively). Please merge chunks with the same prefix before unzipping them (e.g., `cat data.zipchunka* > data.zip && unzip data.zip`).***

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
[data_url]: https://cloud.tsinghua.edu.cn/d/491c750515b94c73bbbb/
[retriever_url]: https://cloud.tsinghua.edu.cn/d/85c358be556e4ce7b8ec/
[rectify_url]: https://cloud.tsinghua.edu.cn/d/71a2177eefb34a33848c/