# FML_2
[Link to Metadata](https://archive.ics.uci.edu/dataset/2/adult)

ToDos:

- [ ] Business/Data Understanding
- [ ] Data Prep - Lisa
  - [x] cleaning e.g. unnecessary columns (or which would be problematic for ethics & data privacy)
  - [ ] test model performance with all columns available?
  - [ ] feature engineering?
  - [x] create centralized test set of all - Fabian
  - [x] create seed for val set for reproducability
  - [ ] create data_prep.py (library, auch für einzel Daten für deployment?)
  - [ ] Optimize Models?
- [ ] Logistic Regression Training - Daniel
  - [x] Centralized - Fabian
  - [x] Individual for each Bank (for comparison to FML)
  - [ ] (Optional) Federated - Fabian (SVC is better Model, but only slightly)
  - [ ] Explainability
- [ ] SVM Training - Daniel 
  - [x] Centralized - Fabian
  - [x] Individual for each Bank (for comparison to FML)
  - [ ] Federated - Fabian
  - [ ] Explainability
- [ ] NN Model Training (TensorFlow) - Vanessa
  - [x] Centralized
  - [x] Individual for each Bank (for comparison to FML) - Fabian
  - [ ] Optimize
  - [ ] Federated - Fabian
  - [ ] Explainability
- [ ] (Optional) Connection to W&B (upload stats)
  - [ ] Explainability
  - [ ] Performance
  - [ ] Eval Accuracy
- [ ] (Optional) Streamlit/Gradio Test Interface für Model = **Deployment**?
- [ ] Code Cleaning
- [ ] Security issues?!
- [ ] Model Card?
- [ ] Präsentation (ans CRISP-DM anpassen)
  - [ ] Results of Business/Data Understanding
  - [ ] Results of Feature Engineering
  - [ ] FL Model Performance
  - [ ] Comparison to centralized Model (per bank?)
  - [ ] Comparison to simpler non-NN Model_(s) (performance & acc)
  - [ ] Performance Criteria (Precision?)
  - [ ] Fairness & Explainability
  - [ ] Security issues?!
  - [ ] Model Card?


Thoughts:
Compare sperately trained Bank Performance with Federated Performance (Global & bank specific data) = What could single bank achieve vs FML?
Compare centrtal model to FML?