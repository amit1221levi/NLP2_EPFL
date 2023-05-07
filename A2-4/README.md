[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-8d59dc4de5201274e310e4c54b9627a8934c3b88527886e3b421487c677d23eb.svg)](https://classroom.github.com/a/Qu0O8OUd)
#  Assignment 2 - Transfer Learning and Data Augmentation 💬

<div style="padding:15px 20px 20px 20px;border-left:3px solid green;background-color:#e4fae4;border-radius: 20px;">

## **Assignment Description**
- In the first part of this assignment, you will need to implement training (fine-tuning) and evaluation of a pre-trained language model ([DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) ), on natural language inference (NLI) task for recognizing textual entailment (RTE).

- Following the first finetuning task, you will need to identify the shortcut (i.e. some salient or toxic features) that the model learnt for the specific task. 

- For part-3, you are supposed to annotate 100 randomly assigned test datapoints as ground-truth labels. Additionally, the cross annotation should be conducted by another one or two annotators, and you will learn about how to calculate the agreement statistics as a significant characteristic reflecting the quality of a collected dataset.

- For part-4, since the human annotation is quite time- and effort-consuming, there are plenty of ways to get silver-labels from automatic labeling to augment the dataset scale. We provide the reference to some simple methods (EDA and Back Translation) but you are encouraged to explore other advanced mechanisms. You will evaluate the improvement of your model performance by using your data augmentation method.

For each part, you will need to complete the code in the corresponding `.py` files (`nli.py` for Part-1, `shortcut.py` for Part-2, `eda.py` for Part-4). You will be provided with the function descriptions and detailed instructions about the code snippet you need to write.


### Table of Contents
- **[PART 1: Model Finetuning for NLI](#1)**
    - [1.1 Data Processing](#11)
    - [1.2 Model Training and Evaluation](#12)
- **[PART 2: Identify Model Shortcut](#2)**
    - [2.1 Word-Pair Pattern Extraction](#21)
    - [2.2 Distill Potentially Useful Patterns](#22)
    - [2.3 Case Study](#23)
- **[PART 3: Annotate New Data](#3)**
    - [3.1 Write an Annotation Guideline](#31)
    - [3.2 Annotate Your 100 Datapoints with Partner(s)](#32)
    - [3.3 Agreement Measure](#33)
    - [3.4 Robustness Check](#34)
- **[PART 4: Data Augmentation](#4)**
    
### Deliverables

- ✅ This jupyter notebook
- ✅ `nli.py` file
- ✅ `shortcut.py` file
- ✅ Finetuned DistilBERT models for NLI task (Part 1 and Part 4)
- ✅ Annotated and cross-annotated data files (Part 3)
- ✅ New dataset from data augmentation (Part 4)

</div>
