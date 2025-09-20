# 🧬 DNA Sequence Classification using Machine Learning
![DNA Sequence Classification](https://github.com/ZahraSahranavard/DNA-Sequence-ML/blob/main/Image/DNA%20Sequence%20Classification.png)

## 🔹 Overview 
This project classifies **DNA sequences** using machine learning techniques. DNA sequence classification is a fundamental task in bioinformatics with applications in genomics, proteomics, and personalized medicine. This project offers a simple and effective approach to DNA sequence classification using **k-mer analysis** and the **Multinomial Naive Bayes** algorithm.

## 📂 Dataset
**The dataset used in this project is sourced from the [DNA-Dataset](https://github.com/ZahraSahranavard/DNA-Dataset) repository. It contains DNA sequences and their corresponding classes, such as human, chimpanzee, dog, etc. The `human_data.txt` file is used for training and testing the model.**

## 🔬 K-mer Analysis
**K-mer analysis** involves breaking down DNA sequences into smaller, overlapping substrings of length `k` (called **k-mers**).  

- Example:  
  DNA sequence: `ATCGTAC`  
  6-mers: `ATCGTA`, `TCGTAC`  

By counting the frequency of different k-mers in a sequence, we can create a **feature vector** that captures the composition of the sequence.  
This approach is effective for DNA sequence classification because different classes of DNA may have **distinct k-mer frequency profiles**.

## ⚙️ Dependencies
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn

You can install these dependencies using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
```

## 🚀 Usage

1. **Clone this repository (project code):**
```bash
git clone https://github.com/ZahraSahranavard/DNA-Sequence-ML
```
```bash
cd DNA-Sequence-ML
```
2. **Clone the dataset repository:**
```bash
git clone https://github.com/ZahraSahranavard/DNA-Dataset.git
```
3. **Run the notebook:**

- Open `DNA-Data-Analysis.ipynb` in Google Colab or Jupyter Notebook.

- Run all cells to train and evaluate the model.

## 📊 Results
The **Multinomial Naive Bayes classifier** shows excellent performance on the dataset:

- **Accuracy**: 0.984  
- **Precision**: 0.984  
- **Recall**: 0.984  
- **F1-score**: 0.984  

These results demonstrate that **k-mer based feature extraction + Naive Bayes** is an effective approach for DNA sequence classification.

## 🔮 Future Work
- **Explore alternative classifiers** such as Support Vector Machines (SVM), Random Forest, and Gradient Boosting to compare their performance with Naive Bayes.  
- **Incorporate DNA-specific embeddings** (e.g., DNA2Vec or k-mer embedding models) to capture deeper biological patterns in nucleotide sequences.  
- **Leverage deep learning architectures** like Convolutional Neural Networks (CNNs) for motif detection, Recurrent Neural Networks (RNNs) for sequence modeling, and Transformers for capturing long-range dependencies in DNA.  

## 📜 License
This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
