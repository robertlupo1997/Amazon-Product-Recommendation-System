# Amazon-Product-Recommendation-System

Welcome to the **Amazon Product Recommendation System** project! This project demonstrates the development of a recommendation system using the Amazon product reviews dataset. The system recommends products to customers based on their previous ratings for other products.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset Details](#dataset-details)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Models and Methodologies](#models-and-methodologies)
- [Results and Observations](#results-and-observations)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Project Overview
Recommender systems are vital tools that help businesses suggest products to users based on their preferences, effectively addressing the problem of information overload. This project focuses on building a recommendation system for Amazon's electronic products.

### Objective
To create a system that recommends products to customers based on their previous ratings using collaborative filtering and matrix factorization techniques.

---

## Dataset Details

We use the **Amazon Electronics Data** from Kaggle. You can download it here:  
[**Amazon Electronics Data**](https://www.kaggle.com/datasets/saurabhbagchi/amazon-electronics-data)

The dataset includes:
- **userId**: Unique identifier for each user.
- **productId**: Unique identifier for each product.
- **rating**: Rating given by a user to a product.
- **timestamp**: Time of the rating (not used in this project).

**Note**: Due to the dataset's large size, users who provided fewer than 50 ratings and products rated fewer than 5 times were excluded to optimize computational efficiency.

---

## Installation
### Dependencies
Ensure the following are installed:
- Python 3.8+
- Jupyter Notebook or Google Colab
- Required Python libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-surprise`

---

## Project Structure
- **Robert_Lupo_Recommendation_Systems_Learner_Notebook_Full_Code.html**: Rendered HTML of the notebook.
- **Robert_Lupo_Recommendation_Systems_Learner_Notebook_Full_Code.ipynb**: The Jupyter Notebook containing the full code and explanations.

---

## Usage
1. Open the Jupyter Notebook or upload it to Google Colab.
2. Run the cells sequentially to:
   - Explore and preprocess the data.
   - Implement recommendation algorithms.
   - Evaluate the models.

---

## Models and Methodologies
### 1. Rank-Based Recommendation System
- Recommends products based on popularity (average ratings and number of interactions).

### 2. Collaborative Filtering
- **User-User Similarity-Based Recommendations**.
- **Item-Item Similarity-Based Recommendations**.

### 3. Matrix Factorization
- **Singular Value Decomposition (SVD)** for latent factor analysis.

### Tools and Libraries
- `scikit-surprise`: Used for building and evaluating recommendation models.
- `matplotlib` and `seaborn`: For visualizing data and results.

---

## Results and Observations
- **Best Performing Model**: Optimized SVD with:
  - RMSE: 0.8808
  - Precision: 85.4%
  - Recall: 87.8%
  - F1 Score: 86.6%
- Collaborative filtering models performed well, with room for improvement in sparsity handling.

---

## Future Enhancements
- Incorporate side information (e.g., product metadata, text reviews) to improve recommendations.
- Experiment with advanced techniques like SVD++, deep learning, or hybrid models.
- Optimize models for real-time performance.

---

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request.

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgments
- Dataset from Amazon product reviews.
- Recommendation algorithms implemented using the `surprise` library.
- Project inspired by the need for personalized user experiences in e-commerce.
