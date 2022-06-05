This branch contains code for finetuning the SOTA retriever trained on NQ with knowledge distilled from a reader, which basically reuses code for the following two papers:
* [Distilling Knowledge from Reader to Retriever for Question Answering][distilled_dpr_repo]
* [Dense Passage Retrieval for Open-Domain Question Answering][dpr_repo]

## Dependencies

1. Set `prefix` to your own path in `DPR.yml`
2. Create the conda environment with the following command
   ```bash
    conda env create -f DPR.yml
    ```

# Data

Suppose the target multi-answer dataset is called `<dataset>`, such as `AmbigQA`, `WebQSP`, or a name of your own dataset.

### Data format

The expected data format is a list of entry examples, where each entry example is a dictionary containing
- `id`: example id
- `question`: question text
- `annotations`: a list of dictionaries, please refer to the definition from the [AmbigQA][ambigqa_repo] repository
- `answers`: list of annotated answer texts which are extracted from `annotations`
- `ctxs`: a list of passages retrieved with the retriever to be finetuned; each item is a dictionary containing
    - `title`: article title
    - `text`: passage text

Entry example:
```
{
  'id': '0',
  'question': 'Who plays the doctor in dexter season 1?',
  'annotations': [
                    {
                        'type': 'singleAnswer', 'answer': ['Tony Goldwyn', 'Goldwyn']
                    }
                 ],
  'answers': ['Tony Goldwyn', 'Goldwyn'],
  'ctxs': [
            {
                "title": "Dexter (season 1)",
                "text": "Dexter (season 1) The first season of \"Dexter\" is an adaptation of Jeff Lindsay's first novel in the \"Dexter\" series, \"Darkly Dreaming Dexter\". Subsequent seasons have featured original storylines. This season aired from October 1, 2006 to December 17, 2006, and follows Dexter's investigation of \"The Ice Truck Killer\". Introduced in the first episode, \"Dexter\", this serial killer targets prostitutes and leaves their bodies severed and bloodless. At the same time, Dexter's foster sister, Debra Morgan (Jennifer Carpenter), a vice squad officer, aspires to work in the homicide department, and Dexter's girlfriend, Rita Bennett (Julie Benz), wants their relationship to"
            },
            {
                "title": "Dexter (TV series)",
                "text": "title credits in season two). Erik King portrayed the troubled Sgt James Doakes for the first two seasons of the show. Desmond Harrington joined the cast in season three as Joey Quinn; his name was promoted to the title credits as of season four. Geoff Pierson plays Captain Tom Matthews of Miami Metro Homicide. Julie Benz starred as Dexter's girlfriend, then wife, Rita in seasons one to four, with a guest appearance in season five. Rita's children, Astor and Cody, are played by Christina Robinson and Preston Bailey (who replaced Daniel Goldman after the first season). Dexter's infant son Harrison"
            }
          ]
}
```

Data with and without retrieved passages should be placed under `data/retrieval_results/distilled_dpr/<dataset>/` and `data/datasets/<dataset>/`, respectively.

# Retriever finetuning

1. Before finetuning a retriever, `data/` should also include the pre-trained checkpoint and the corresponding wikipedia dump.

2. The following command will finetune the prepared checkpoint, generate passage embeddings for the whole wikipedia dump, and retrieve passages for your target dataset with the finetuned retriever, which will be saved under `DPR_and_ExtractiveReader/outputs/<dataset>/retriever_checkpoints`, `DPR_and_ExtractiveReader/outputs/<dataset>/wikipedia_embeddings`, and `data/retrieval_results/finetuned_distilled_dpr/<dataset>/`.
    ```bash
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python finetune_distilled_dpr.py --dataset <dataset>
    ```

3. Now you are ready to train a recall-then-verify system. Please switch to the branch called `rectify`.

[distilled_dpr_repo]: https://github.com/facebookresearch/FiD
[dpr_repo]: https://github.com/facebookresearch/DPR
[ambigqa_repo]: https://github.com/shmsw25/AmbigQA
