# movie-review-sentiment-analysis-pos-neg
NLP pipeline classifying movie reviews as positive/negative using Bag-of-Words + Naive Bayes (84% test accuracy).

## Overview
- **Task**: Binary sentiment classification (pos/neg) on 40,000 movie reviews
- **Approach**: NLTK preprocessing → Bag-of-Words (CountVectorizer) → Multinomial Naive Bayes
- **Results**: 88% training accuracy, 84% accuracy on held-out test data

## Pipeline
1. Text cleaning (lowercase, HTML tag removal)
2. Tokenization (RegexpTokenizer)
3. Stopword removal (NLTK)
4. Stemming (PorterStemmer)
5. Vectorization (CountVectorizer, 30,000-feature vocabulary)
6. Classification (Multinomial Naive Bayes)

## Results
| Metric | Score |
|--------|-------|
| Train Accuracy | 88% |
| Test Accuracy  | 84% |

## Tech Stack
Python, Pandas, NumPy, NLTK, Scikit-learn

## Usage
\`\`\`bash
pip install -r requirements.txt
jupyter notebook notebook.ipynb
\`\`\`

## Future Improvements
- Compare against TF-IDF vectorization
- Try Logistic Regression / SVM for comparison
- Experiment with bigrams
