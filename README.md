# DNA Sequence Classification using Machine Learning

This project classifies DNA sequences using machine learning techniques. DNA sequence classification is a fundamental task in bioinformatics with applications in genomics, proteomics, and personalized medicine. This project offers a simple and effective approach to DNA sequence classification using k-mer analysis and the Multinomial Naive Bayes algorithm.

## Dataset
**The dataset used in this project is sourced from the [DNA-Dataset](https://github.com/ZahraSahranavard/DNA-Dataset) repository. It contains DNA sequences and their corresponding classes, such as human, chimpanzee, dog, etc. The `human_data.txt` file is used for training and testing the model.**

## K-mer Analysis
K-mer analysis involves breaking down DNA sequences into smaller, overlapping sequences of length k (called k-mers). By counting the frequency of different k-mers in a sequence, we can create a feature vector that captures the composition of the sequence. This approach is effective for DNA sequence classification because different classes of DNA may have distinct k-mer frequency profiles.

## Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn