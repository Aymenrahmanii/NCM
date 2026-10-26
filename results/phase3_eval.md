# Phase 3: official NCM checkpoints on an RTX 3050 (first 50 test examples)

| task | precision | params | weights MB | peak MB | ms/example | score |
|---|---|---|---|---|---|---|
| xsum | fp32 | 88.0M | 691 | 1498 | 436.1 | R-1 32.76 / R-2 12.16 / R-L 26.47 |
| wmt14 | fp32 | 120.0M | 1086 | 1831 | 109.0 | BLEU 15.28 |
| xsum | fp16 | 88.0M | 698 | 1694 | 174.4 | R-1 32.84 / R-2 12.16 / R-L 26.55 |
| wmt14 | fp16 | 120.0M | 1086 | 2454 | 58.7 | BLEU 14.96 |
| xsum | bf16 | 88.0M | 698 | 1694 | 168.4 | R-1 32.64 / R-2 12.43 / R-L 26.36 |
| wmt14 | bf16 | 120.0M | 1086 | 2454 | 51.9 | BLEU 14.35 |
| xsum (raw text input) | fp32 | 88.0M | 698 | 1497 | 446.0 | R-1 32.76 / R-2 12.16 / R-L 26.47 |

Paper: XSum ROUGE-L 28.0 (full test set), WMT14 BLEU 22.6 (full test set).

## Official evaluate.py: xsum (n=50)

```
[evaluate] 1 shard(s) on GPU(s) 0 -> /home/aymen/ncm-lab/artifacts/official_eval/xsum/

merged 1 shards -> 50 xsum outputs

    R-1     R-2     R-L   genPPL   words
  32.76   12.16   26.48    91.40    20.9

Loading weights:   0%|          | 0/148 [00:00<?, ?it/s]
Loading weights:   1%|▏         | 2/148 [00:00<00:12, 12.08it/s]
Loading weights:   5%|▌         | 8/148 [00:00<00:05, 23.95it/s]
Loading weights:  11%|█         | 16/148 [00:00<00:04, 32.29it/s]
Loading weights:  22%|██▏       | 32/148 [00:00<00:01, 65.27it/s]
Loading weights:  36%|███▋      | 54/148 [00:00<00:01, 93.98it/s]
Loading weights:  66%|██████▌   | 98/148 [00:00<00:00, 171.24it/s]
Loading weights: 100%|██████████| 148/148 [00:00<00:00, 159.01it/s]
```

## Official evaluate.py: wmt14 (n=50)

```
[evaluate] 1 shard(s) on GPU(s) 0 -> /home/aymen/ncm-lab/artifacts/official_eval/wmt14/

merged 1 shards -> 50 wmt14 outputs

    BLEU  BLEU(lc)    chrF  len ratio
   15.28     16.01   42.82      0.880

BLEU|True = 15.28 56.0/24.6/12.0/6.1 (BP = 0.857 ratio = 0.867 hyp_len = 1079 ref_len = 1245)
```

## Official evaluate.py: sudoku-easy (n=200)

```
[evaluate] 1 shard(s) on GPU(s) 0 -> /home/aymen/ncm-lab/artifacts/official_eval/sudoku-easy/

merged 1 shards -> 200 sudoku-easy outputs

 solved %  solved  cell %  valid %
    99.00  198/200   99.91    99.00
```

## Official evaluate.py: sudoku-medium (n=200)

```
[evaluate] 1 shard(s) on GPU(s) 0 -> /home/aymen/ncm-lab/artifacts/official_eval/sudoku-medium/

merged 1 shards -> 200 sudoku-medium outputs

 solved %  solved  cell %  valid %
    97.50  195/200   99.91    97.50
```

## Official evaluate.py: sudoku-hard (n=200)

```
[evaluate] 1 shard(s) on GPU(s) 0 -> /home/aymen/ncm-lab/artifacts/official_eval/sudoku-hard/

merged 1 shards -> 200 sudoku-hard outputs

 solved %  solved  cell %  valid %
    84.50  169/200   98.35    84.50
```

## Examples (fp32)

- **xsum** output: The number of beds made available for homeless people living in accommodation in Wales has risen in last year, a Welsh housing homelessness charity has said.
  reference: There is a "chronic" need for more housing for prison leavers in Wales, according to a charity.
- **xsum** output: A man has been arrested in connection with a number of firearms ammunition and ammunition were seized in Edinburgh.
  reference: A man has appeared in court after firearms, ammunition and cash were seized by police in Edinburgh.
- **xsum** output: Four people have been charged with, kidnapping, beating, beat and beat a white black man in Chicago.
  reference: Four people accused of kidnapping and torturing a mentally disabled man in a "racially motivated" attack streamed on Facebook have been denied bail.
- **xsum** output: West Bromwich Albion have confirmed director as Reading first-team coach at the end of the season with the Championship club.
  reference: West Brom have appointed Nicky Hammond as technical director, ending his 20-year association with Reading.
- **xsum** output: A diet that produces causes fat in the pancreas is limbs and reverse the results of type 2 diabetes, a US study suggests.
  reference: The pancreas can be triggered to regenerate itself through a type of fasting diet, say US researchers.
- **wmt14** output: Gach: Even more safety for.
  reference: Gutach: Increased safety for pedestrians
- **wmt14** output: They are not 100 meters away from another another: On Tuesday in the new B 33 pedestrian traffic was moved into operation in the village premises - in sight of the older town hall lights.
  reference: They are not even 100 metres apart: On Tuesday, the new B 33 pedestrian lights in Dorfparkplatz in Gutach became operational - within view of the existing Town Hall traffic lights.
- **wmt14** output: There are so two so close: intention or coup d coup coup?
  reference: Two sets of lights so close to one another: intentional or just a silly error?
- **wmt14** output: This answered this question that could very clearly clearly yesterday.
  reference: Yesterday, Gutacht's Mayor gave a clear answer to this question.
- **wmt14** output: "We Rathausaus was installed at the time, because it ensures the school path explains says Eckert yesterday.
  reference: "At the time, the Town Hall traffic lights were installed because this was a school route," explained Eckert yesterday.
