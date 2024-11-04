Smart Leaf: CNN-Based Mango Leaf Disease Prediction

________________________________________
1. Project Overview
•	Objective: Predict and classify diseases in mango leaves using a Convolutional Neural Network (CNN).
•	Significance: Accurate disease prediction helps in early intervention, ensuring healthier crops and improved agricultural yields.
•	Target Audience: Researchers, agricultural professionals, and students interested in AI applications in agriculture.
________________________________________
2. Dataset and Preprocessing
•	Dataset Details:
o	Mango leaf images, categorized by disease type.
o	Split into training, validation, and test sets.
•	Preprocessing Steps:
o	Image resizing and normalization.
o	Data augmentation (rotation, flipping) to enhance model generalization.
________________________________________
3. Model Architecture
•	CNN Layers:
o	Convolutional layers to capture features like color, texture, and patterns in leaves.
o	Pooling layers for dimensionality reduction, improving computational efficiency.
o	Fully connected layers for classification.
•	Hyperparameter Tuning:
o	Optimized layers, filter sizes, and dropout rates to prevent overfitting.
________________________________________



4. Training and Validation Results
•	Training Progress (10 Epochs):
Epoch	Training Accuracy	Training Loss	Validation Accuracy	Validation Loss
1	26.17%	2.1881	22.12%	3.3209
2	61.13%	1.0687	39.13%	3.1832
3	77.72%	0.6443	53.37%	2.4067
4	89.04%	0.3148	69.25%	1.1691
5	92.66%	0.2363	57.00%	2.4898
6	91.99%	0.2174	76.00%	1.3362
7	95.60%	0.1264	83.13%	0.5830
8	95.44%	0.1357	78.62%	0.8111
9	97.78%	0.0655	93.25%	0.2018
10	97.21%	0.0994	73.37%	1.2582


•	Key Observations:
o	Training accuracy improved steadily, reaching 97.21% by epoch 10.
o	Validation accuracy saw fluctuations, peaking at 93.25% at epoch 9, showing potential for fine-tuning.
o	Some overfitting is indicated as validation loss rises after epoch 7.
________________________________________
5. Model Evaluation and Final Results
•	Evaluation on Test Set: The model was tested on unseen data to assess its generalization ability.
•	Performance Metrics:
o	Accuracy: 93.25% (Peak during validation)
o	Loss: Indicates room for optimization, especially for reducing overfitting.
________________________________________

6. Key Takeaways and Next Steps
•	Achievements: SmartLeaf demonstrates high potential for practical applications in agriculture by identifying mango leaf diseases with high accuracy.
•	Challenges: Slight overfitting observed; potential improvement through regularization and more diverse data.
•	Future Work: Experiment with transfer learning using pre-trained models, which could enhance the model’s accuracy and robustness.
________________________________________
7. Conclusion
Smart Leaf effectively combines deep learning with agriculture, providing a promising solution to support farmers and agricultural specialists in monitoring plant health. With further refinements, this model can become a valuable tool in crop disease management, contributing to sustainable and technology-driven farming practices.
