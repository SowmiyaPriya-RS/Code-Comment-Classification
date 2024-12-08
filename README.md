# NLBSE Tool Competition - Code Comment Classification

## Overview

This project is developed for the NLBSE Tool Competition, where the task is to classify code comments into predefined categories. The model uses a pre-trained BERT model from Hugging Face for fine-tuning on the provided dataset. The dataset consists of code comments in Java, Python, and Pharo, with each language having different categories for classifying the comments.

## Project Setup

1. **Install Requirements**

   - Python 3.x
   - PyTorch 2.x
   - Hugging Face Transformers
   - Datasets library
   - scikit-learn
   - tqdm
   - pandas
   - numpy

2. **Download the Dataset**

   The dataset is available on Hugging Face and consists of six splits, two (train and test) per language. Each row represents a sentence with the following columns:
   - `class`: Class name of the source code file.
   - `comment_sentence`: The actual sentence within a class comment.
   - `partition`: Split identifier (0 for training, 1 for testing).
   - `combo`: Class name appended to the sentence.
   - `labels`: Ground-truth binary list of categories the comment belongs to.

   Categories for different languages:
   - **Java**: [summary, Ownership, Expand, usage, Pointer, deprecation, rational]
   - **Python**: [Usage, Parameters, DevelopmentNotes, Expand, Summary]
   - **Pharo**: [Keyimplementationpoints, Example, Responsibilities, Classreferences, Intent, Keymessages, Collaborators]

3. **Fine-tuning the Pre-trained BERT Model**

   To fine-tune the pre-trained BERT model for classification:
   
   - **Load a Pre-trained BERT Model**: Use a pre-trained BERT model from Hugging Face. The `bert-base-uncased` model is commonly used for this task.
   - **Load and Tokenize the Dataset**: Load the dataset and apply tokenization using the pre-trained BERT tokenizer. Ensure the sentences are padded or truncated to a consistent length and that they are tokenized correctly for BERT.
   - **Set Up Training Arguments**: Define the hyperparameters for training, such as the learning rate, batch size, number of epochs, and evaluation strategy.
   - **Train the Model**: Use the Hugging Face `Trainer` class to handle the training and evaluation of the model. Specify the training and evaluation datasets.


4. **Evaluation and Results**

   After training the model, you can evaluate its performance on the test set. The evaluation metric can be accuracy or F1 score, depending on your preference.

   ```python
   results = trainer.evaluate()
   print(results)
   ```

## Results

Here are the evaluation results for the different languages and categories:

| Language | Category | Precision | Recall | F1 Score |
|----------|----------|-----------|--------|----------|
| **Java** | summary | 0.7865 | 0.8632 | 0.8231 |
| **Java** | Ownership | 0.9783 | 1.0000 | 0.9890 |
| **Java** | Expand | 0.2051 | 0.0784 | 0.1135 |
| **Java** | usage | 0.9607 | 0.6798 | 0.7962 |
| **Java** | Pointer | 0.7380 | 0.9185 | 0.8184 |
| **Java** | deprecation | 0.0000 | 0.0000 | 0.0000 |
| **Java** | rational | 0.1008 | 0.1765 | 0.1283 |
| **Python** | Usage | 0.7586 | 0.5455 | 0.6346 |
| **Python** | Parameters | 0.6327 | 0.4844 | 0.5487 |
| **Python** | DevelopmentNotes | 0.0000 | 0.0000 | 0.0000 |
| **Python** | Expand | 0.3182 | 0.1094 | 0.1628 |
| **Python** | Summary | 0.5631 | 0.7073 | 0.6270 |
| **Pharo** | Keyimplementationpoints | 0.0000 | 0.0000 | 0.0000 |
| **Pharo** | Example | 0.9048 | 0.6387 | 0.7488 |
| **Pharo** | Responsibilities | 0.4878 | 0.7692 | 0.5970 |
| **Pharo** | Classreferences | 0.0000 | 0.0000 | 0.0000 |
| **Pharo** | Intent | 0.8710 | 0.9000 | 0.8852 |
| **Pharo** | Keymessages | 0.0000 | 0.0000 | 0.0000 |
| **Pharo** | Collaborators | 0.0000 | 0.0000 | 0.0000 |

## References

Please cite the following works when using the dataset or referring to the original work behind this competition:

```bibtex
@inproceedings{nlbse2025,
  author={Colavito, Giuseppe and Al-Kaswan, Ali and Stulova, Nataliia and Rani, Pooja},
  title={The NLBSE'25 Tool Competition},
  booktitle={Proceedings of The 4th International Workshop on Natural Language-based Software Engineering (NLBSE'25)},
  year={2025}
}

@article{rani2021,
  title={How to identify class comment types? A multi-language approach for class comment classification},
  author={Rani, Pooja and Panichella, Sebastiano and Leuenberger, Manuel and Di Sorbo, Andrea and Nierstrasz, Oscar},
  journal={Journal of systems and software},
  volume={181},
  pages={111047},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{pascarella2017,
  title={Classifying code comments in Java open-source software systems},
  author={Pascarella, Luca and Bacchelli, Alberto},
  booktitle={2017 IEEE/ACM 14th International Conference on Mining Software Repositories (MSR)},
  year={2017},
  organization={IEEE}
}

@inproceedings{alkaswan2023stacc,
  title={Stacc: Code comment classification using sentencetransformers},
  author={Al-Kaswan, Ali and Izadi, Maliheh and Van Deursen, Arie},
  booktitle={2023 IEEE/ACM 2nd International Workshop on Natural Language-Based Software Engineering (NLBSE)},
  pages={28--31},
  year={2023},
  organization={IEEE}
}
```
## Contributors

- **Aishwarya Senthilkumar** - [aishwarya-senthilkumar](https://github.com/aishwarya-senthilkumar)
- **Sowmiya Priya Ramachandran Senthilkumar** - [SowmiyaPriya-RS](https://github.com/SowmiyaPriya-RS)
