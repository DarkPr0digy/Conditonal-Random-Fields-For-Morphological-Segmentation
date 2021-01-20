# Morphological Segmentation of Low Resource Languages using Conditional Random Fields
## Computer Science Honours Project
The Nguni languages of Southern Africa have had two types
of Language Processing Applications Developed for them:
Rule-based applications which have high accuracy whilst
only working for a small subset of data, and Data-driven
applications that tend to be low accuracy. We aim to improve
the data-driven approaches. How can we facilitate the development of more high accuracy applications? Within the field
of Computational Linguistics there is a study called Morphological Segmentation. This study deals with the breaking
down of individual words into morphemes, the smallest unit
of language with meaning. The use of morphological segmentation may facilitate the creation of better language processing applications because with little data, words could be
broken down to their smallest units making it easier to determine their meaning. Low resource languages are languages
for which there is little access to text data, both annotated and unannotated. 
Thus a model is needed that can segment and label words with high accuracy, and do that with low amounts of text data.
This task can be accomplished through the use of several Machine Learning models but the model that will be the focus of this paper is known as Conditional Random Fields (CRF). 
CRFs are a class of discriminative probabilistic models used to label sequence data such as text corpora and images,
using Supervised Machine Learning.
Two varieties of CRFs were implemented to this end: a traditional log-linear CRF and a Bidirectional Long Short Term Memory CRF. 
After the implementation of both models we were able to observe average F1 scores of 98% in the segmentation task across both models. Due to a high variance it would not be representative to give the average score for
the other two tasks, however, it should be noted that the Baseline CRF consistently outperformed the implemented Bidirectional Long Short Term Memory CRF. \
\
The baseline CRF was implemented using the python library sklearn_crfsuite which is accessible at: https://sklearn-crfsuite.readthedocs.io/en/latest/ \
The bi_lstm_crf class is an edited version of bi_lstm_crf made by username: jidasheng on github which can be found at: https://github.com/jidasheng/bi-lstm-crf
