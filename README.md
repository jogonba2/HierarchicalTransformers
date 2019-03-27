# HierarchicalTransformers
Hierarchical Transformers for Extractive Text Summarization

Sentence Encoder Layer 1 Self Attentions
![alt text](https://github.com/jogonba2/HierarchicalTransformers/blob/master/SentenceAttention-Layer1.PNG)

Sentence Encoder Layer 2 Self Attentions
![alt text](https://github.com/jogonba2/HierarchicalTransformers/blob/master/SentenceAttention-Layer2.PNG)

Sentence Encoder Layer 2 Avg Self Attentions
![alt text](https://github.com/jogonba2/HierarchicalTransformers/blob/master/AvgHeadAttention-Layer2.png)


Sentence Encoder Layer 2 Avg Self Attentions Sum for Sentences (Final sentence relevance)
![alt text](https://github.com/jogonba2/HierarchicalTransformers/blob/master/SumSentenceAvgHeadAttention-Layer2.png)



HTS_3 (FULL LENGTH ROUGE F1)
---------------------------------------------
0 ROUGE-1 Average_R: 0.53533 (95%-conf.int. 0.53247 - 0.53828)
0 ROUGE-1 Average_P: 0.30937 (95%-conf.int. 0.30710 - 0.31156)
0 ROUGE-1 Average_F: 0.37714 (95%-conf.int. 0.37510 - 0.37912)
---------------------------------------------
0 ROUGE-2 Average_R: 0.23966 (95%-conf.int. 0.23693 - 0.24235)
0 ROUGE-2 Average_P: 0.13721 (95%-conf.int. 0.13541 - 0.13902)
0 ROUGE-2 Average_F: 0.16768 (95%-conf.int. 0.16568 - 0.16965)
---------------------------------------------
0 ROUGE-L Average_R: 0.48730 (95%-conf.int. 0.48451 - 0.49011)
0 ROUGE-L Average_P: 0.28191 (95%-conf.int. 0.27972 - 0.28405)
0 ROUGE-L Average_F: 0.34356 (95%-conf.int. 0.34152 - 0.34554)

LEAD_3 (FULL LENGTH ROUGE F1)

---------------------------------------------
0 ROUGE-1 Average_R: 0.52032 (95%-conf.int. 0.51744 - 0.52319)
0 ROUGE-1 Average_P: 0.32515 (95%-conf.int. 0.32294 - 0.32742)
0 ROUGE-1 Average_F: 0.38625 (95%-conf.int. 0.38411 - 0.38836)
---------------------------------------------
0 ROUGE-2 Average_R: 0.23259 (95%-conf.int. 0.22967 - 0.23541)
0 ROUGE-2 Average_P: 0.14417 (95%-conf.int. 0.14220 - 0.14608)
0 ROUGE-2 Average_F: 0.17164 (95%-conf.int. 0.16947 - 0.17370)
---------------------------------------------
0 ROUGE-L Average_R: 0.47464 (95%-conf.int. 0.47180 - 0.47750)
0 ROUGE-L Average_P: 0.29688 (95%-conf.int. 0.29477 - 0.29902)
0 ROUGE-L Average_F: 0.35256 (95%-conf.int. 0.35038 - 0.35463)


SHANN_3 (FULL LENGTH ROUGE F1)

---------------------------------------------
0 ROUGE-1 Average_R: 0.55350 (95%-conf.int. 0.55073 - 0.55643)
0 ROUGE-1 Average_P: 0.29392 (95%-conf.int. 0.29187 - 0.29604)
0 ROUGE-1 Average_F: 0.36922 (95%-conf.int. 0.36729 - 0.37117)
---------------------------------------------
0 ROUGE-2 Average_R: 0.25110 (95%-conf.int. 0.24839 - 0.25386)
0 ROUGE-2 Average_P: 0.13217 (95%-conf.int. 0.13054 - 0.13399)
0 ROUGE-2 Average_F: 0.16630 (95%-conf.int. 0.16447 - 0.16821)
---------------------------------------------
0 ROUGE-L Average_R: 0.50345 (95%-conf.int. 0.50085 - 0.50613)
0 ROUGE-L Average_P: 0.26782 (95%-conf.int. 0.26591 - 0.26986)
0 ROUGE-L Average_F: 0.33625 (95%-conf.int. 0.33430 - 0.33813)
