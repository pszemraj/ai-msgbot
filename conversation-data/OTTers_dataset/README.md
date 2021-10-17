# 🦦 OTTers: One-turn Topic Transitions for Open-Domain Dialogue

Explanation of data collection, data splits, and experiments can be found [here](https://arxiv.org/abs/2105.13710).

## Data
You can download the data already divided in the two `in-domain` and `out-of-domain` splits.


Number of topic pairs for each data split:
- **in-domain**: allows one of the topics in each pair of the `test` set to appear in the `train` set, although with a different second topic,
- **out-of-domain**: none of the topics in the `test` set are present in any of the topic-pairs in the `train` set.

|                   | Train | Dev | Test |
| ----              | ----  |    ----     |    ----    | 
| `in-domain`    | 1929  |    1160  |    1158     |
| `out-of-domain`  | 2034 |   1152    |    1130   |   

-------------

## Models
To reproduce our results clone [Multigen](https://github.com/cdjhz/multigen) and follow their instructions to train the model on $\alpha$NLG, then save the trained model as `anlg`.    
Preprocess multi-hop relational paths for OTTers. Set `$DATA` to either be `in_domain` or `out_of_domain`.
```bash
export DATA=in_domain
python ground_concepts_simple.py $DATA
python find_neighbours.py $DATA
python filter_triple.py $DATA
```
### Training
```bash
export DATA_TYPE={in_domain, out_of_domain}
export ROOT_PATH=..
export DEVICE=1
export PRE_TRAINED={gpt2-small, anlg}
CUDA_VISIBLE_DEVICES=${DEVICE} \
python3 main.py \
--train_data_file ${ROOT_PATH}/data/${DATA_TYPE}/train \
--dev_data_file ${ROOT_PATH}/data/${DATA_TYPE}/dev \
--test_data_file ${ROOT_PATH}/data/${DATA_TYPE}/test \
--graph_path 2hops_100_directed_triple_filter.json \
--output_dir ${ROOT_PATH}/models/${DATA_TYPE}/grf-${DATA_TYPE} \
--source_length 32 \
--target_length 16 \
--model_type gpt2 \
--model_name_or_path ${ROOT_PATH}/models/${PRE_TRAINED} \
--do_train \
--per_gpu_train_batch_size 16 \
--per_gpu_eval_batch_size 16 \
--workers 7 \
--seed 42 \
--evaluate_metrics bleu \
--overwrite_output_dir \
--num_train_epochs 3 \
--learning_rate 3e-5 \
--aggregate_method max \
--alpha 3 \
--beta 5 \
--gamma 0.5 \
--weight_decay 0.0 \
--warmup_ratio 0.0 \
--logging_steps 20 \
```


---------------
## Citation
If you use the dataset in your work please cite with the following

```
@inproceedings{sevegnani-etal-2021-otters,
    title = "{OTT}ers: One-turn Topic Transitions for Open-Domain Dialogue",
    author = "Sevegnani, Karin  and
      Howcroft, David M.  and
      Konstas, Ioannis  and
      Rieser, Verena",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics"
}
```

