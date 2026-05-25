# Attachment Style Prediction Using Machine Learning

This project explores whether attachment styles can be predicted using a combination of structured psychological survey data and AI-generated relationship reflections.

The analysis combines traditional machine learning techniques with Natural Language Processing (NLP) embeddings to classify four attachment styles:

* Anxious
* Dismissive Avoidant
* Fearful Avoidant
* Secure

## Project Overview

* Generated synthetic first-person relationship reflections using OpenAI based on structured survey responses
* Converted generated text into embeddings using `sentence-transformers/all-MiniLM-L6-v2`
* Compared multiple modelling approaches:

  * Structured data only
  * Text embeddings only
  * Combined structured + text embeddings
    
* Evaluated Logistic Regression, Random Forest and XGBoost classifiers
* Selected a tuned XGBoost model as the final model based on overall performance

## Key Results

* Final model accuracy: ~58%
* Macro F1 score: ~57%
* Multiclass AUC: ~0.84
* Fearful Avoidant attachment style achieved the strongest predictive performance

## Potential Applications

* Dating app personalisation
* Relationship coaching tools
* Behavioural segmentation
* Mental wellbeing support systems

## Notes

Synthetic reflections were used as an experimental augmentation technique and do not represent real user-generated text.
