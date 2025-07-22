# BioASQ Medical Question Answering - Neural Network Comparison Study

## Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/BERT-FF6F00?style=for-the-badge&logo=google&logoColor=white" alt="BERT"/>
  <img src="https://img.shields.io/badge/Transformers-FF6F00?style=for-the-badge&logo=huggingface&logoColor=white" alt="Transformers"/>
  <img src="https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-learn"/>
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas"/>
  <img src="https://img.shields.io/badge/Jupyter-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter"/>
</p>

## Project Overview

This project explores three different neural network approaches for medical question answering using the BioASQ biomedical dataset. As someone working in AI for healthcare, I wanted to understand which architecture works best for helping medical professionals find relevant information from scientific literature.

The study compares:
- **Siamese Neural Network** - Basic similarity matching approach
- **LSTM Network** - Sequential text understanding 
- **BERT Transformer** - Advanced contextual analysis

## Why This Matters

Healthcare professionals often need to quickly find specific answers from thousands of research papers. This project demonstrates how different AI approaches can automate this process, potentially saving hours of manual literature review.

## Key Results

| Model | F1-Score | Best Use Case |
|-------|----------|---------------|
| Siamese Network | 0.23 | Basic similarity screening |
| **LSTM Network** | **0.42** | **Production deployment** |
| BERT Transformer | 0.41 | Research applications |

The LSTM approach achieved an **83% improvement** over the baseline, making it the most practical choice for real-world implementation.

## What I Learned

### Technical Insights
- **LSTM was the sweet spot**: Better performance than Siamese networks without the computational overhead of BERT
- **Sequence length matters**: Found that 75 tokens capture 99% of medical sentences effectively
- **Class balancing is crucial**: Reduced dataset from 26k to 11k samples per class for better training

### Practical Considerations
- BERT's marginal improvement (F1: 0.41 vs 0.42) doesn't justify the 3x computational cost
- Medical text has unique characteristics that benefit from sequential understanding
- Production systems need to balance accuracy with response time

## Project Structure

```
├── A3_2_forcollab_(1).ipynb    # Main analysis notebook
├── data/
│   ├── training.csv            # Training dataset
│   ├── dev_test.csv           # Validation dataset
│   └── test.csv               # Test dataset
└── README.md                  # This file
```

## Getting Started

### Prerequisites
```bash
pip install tensorflow transformers scikit-learn pandas numpy matplotlib jupyter
```

### Running the Project
1. Clone this repository
2. Install dependencies
3. Open `A3_2_forcollab_(1).ipynb` in Jupyter
4. Run cells sequentially to reproduce results

## Technical Approach

### 1. Siamese Neural Network
- Uses TF-IDF vectorization with triplet loss
- Computes similarity between questions and answer sentences
- Fast inference but limited understanding of context

### 2. LSTM Network (Recommended)
- 35-dimensional embeddings with 32 LSTM units
- Three dense layers [128, 64, 32] for classification
- Balances performance with computational efficiency

### 3. BERT Transformer
- Pre-trained BERT-base-uncased model
- Fine-tuned with 1-2 additional transformer layers
- Highest sophistication but resource-intensive

## Dataset Details

- **Source**: BioASQ biomedical question answering challenge
- **Content**: Medical questions paired with relevant/irrelevant sentences from PubMed
- **Size**: ~38k training samples after preprocessing
- **Domain**: Covers various medical specialties and research areas

## Real-World Applications

This work has practical applications in:
- **Clinical decision support** - Helping doctors find relevant research quickly
- **Medical education** - Training tools for students and residents  
- **Research acceleration** - Automated literature screening
- **Healthcare chatbots** - Intelligent medical information retrieval

## Future Improvements

While this project shows promising results, there are several areas for enhancement:
- Ensemble methods combining multiple approaches
- Domain-specific BERT fine-tuning on larger medical corpora
- Integration with clinical workflow systems
- Real-time API development for healthcare applications

## Lessons for Healthcare AI

Working on this project reinforced several important principles:
1. **Context matters more than complexity** - LSTM's sequential understanding was more valuable than BERT's sophistication
2. **Production constraints are real** - The best model on paper isn't always the best in practice
3. **Medical data requires special handling** - Domain expertise is crucial for preprocessing and validation
4. **Performance improvements compound** - Even modest accuracy gains can significantly impact clinical workflows

## Connect

Feel free to reach out if you're interested in healthcare AI or have questions about this project. I'm always excited to discuss applications of ML in medical settings.

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue.svg)](https://linkedin.com/in/gishor-thavakumar)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black.svg)](https://github.com/tgishor)

---

*This project demonstrates practical neural network applications in healthcare, showing how thoughtful model selection can create real value for medical professionals.*
