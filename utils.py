import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report

import shap

seed = 42

"""
Function that prints permutation importances and plots the permutation importance plot

perm_imp_train: permutation importance for train data
perm_imp_test: permutation importance for test data
cols: columns of the dataset
plot: True if figure is to be plotted, otherwise False
save_plot: True if figure to be saved, otherwise False
save_path: path to save plot

returns: nothing
"""
def print_permutation_importances(perm_imp_train, perm_imp_test, cols, plot=True, save_plot=True, save_path="fig.png"):
	print("Train permutation importance ")
	for i, feature in enumerate(cols):
		print(f"- {feature} => Mean: {perm_imp_train.importances_mean[i]:.4f} ; Std: {perm_imp_train.importances_std[i]:.4f}")
	
	print("\n Test permutation importance")
	for i, feature in enumerate(cols):
		print(f"- {feature} => Mean: {perm_imp_test.importances_mean[i]:.4f} ; Std: {perm_imp_test.importances_std[i]:.4f}")
	
	if plot: 
		plt.figure(figsize=(12, 6))

		plt.subplot(1, 2, 1)
		plt.bar(range(len(cols)), perm_imp_train.importances_mean)
		plt.xticks(range(len(cols)), cols, rotation=90)
		plt.title("Permutation Feature Importance in train set")

		plt.subplot(1, 2, 2)
		plt.bar(range(len(cols)), perm_imp_test.importances_mean)
		plt.xticks(range(len(cols)), cols, rotation=90)
		plt.title("Permutation Feature Importance in test set")
		if save_plot:
			plt.savefig(save_path, dpi=300, bbox_inches='tight')  # high quality

		plt.show()

"""
Function that prints the accuracies and the accuracy report

model: random forest model
X_train: x training data
Y_train: y training data
X_test: x testing data
Y_test: y testing data
Y_pred: model's predictions on Y_test

returns: Nothing
"""
def print_accuracy(model, X_train, Y_train, X_test, Y_test, Y_pred):
	print(f"Train accuracy: {model.score(X_train, Y_train)}")
	print(f"Test accuracy: {model.score(X_test, Y_test)}")

	report = classification_report(y_true=Y_test, y_pred=Y_pred)
	print(report)

"""
Function that retrieves top k values of a ranking from the dataset

X_test: X testing data
Y_test: Y testing data
Y_pred: model's Y predictions
ranking: a list of indices representing the ranked order of items in 'X_test'.
probs: the prediction probabilities of X_test
k: number of selected elements

returns: a dataframe with the top k elements, their true label, their predicted label and their predicted probability
"""
def get_topK(X_test, Y_test, Y_pred, ranking, probs, k):
	#sorts according to ranking
	x_test_ranked = X_test.iloc[ranking]
	y_test_ranked = Y_test.iloc[ranking]
	probs_ranked = probs[ranking]
	y_pred_ranked = Y_pred[ranking]

	#top k
	x_topk = x_test_ranked.iloc[:k]
	y_topk = y_test_ranked.iloc[:k]
	probs_topk = probs_ranked[:k]
	y_pred_ranked_k = y_pred_ranked[:k]
	
	#concat into a df
	topk_df = x_topk.copy()
	topk_df['true_label'] = y_topk
	topk_df['pred_label'] = y_pred_ranked_k
	topk_df['pred_prob'] = probs_topk

	return topk_df


"""
Function that fits the model to the data and calculates the predictions of Y_test. It also prints the accuracy of the new trained model, 
calculates and prints/plots the permutation importances. Then, it calculates the prediction probabilities and extracts the ranking of the elements. 

model: random forest model
X_train_enc: x train data that is encoded and processed
Y_train: y values of train data
X_test_enc: encoded x test data
Y_test: y values of test data

returns: rankings, the prediction probabilities and the models Y predictions
"""
def fit_and_rank(model, X_train_enc, Y_train, X_test_enc, Y_test):
	# Fit to new data
	model.fit(X_train_enc, Y_train)
	Y_pred = model.predict(X_test_enc)
		
	# Check accuracy
	print_accuracy(model, X_train_enc, Y_train, X_test_enc, Y_test, Y_pred)
	
	#Ranking
	probs = model.predict_proba(X_test_enc)[:, 1]
	ranking = probs.argsort()[::-1]

	return ranking, probs, Y_pred
	

"""
Function that maps the dataset's target column to 0 if the customer is still existing and 1 if it is an attrited customer

df: dataframe of dataset

returns: mapped column
"""
def attrition_flag_map(df):
	return df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

"""
Function that sets up the preprocessing column transformer and returns it. 
The preprocessing involves standard scaling, one hot encoding and ordinal encoding of
appropriate columns based on dataset

returns: preprocessing pipe column transformer
"""
def preprocessing_column_transformer():	
	cat_cols = ["Gender", "Marital_Status"]
	ordinal_cols = ["Education_Level", "Income_Category", "Card_Category"]
	num_cols = [
			"Customer_Age", 
			"Dependent_count", 
			"Months_on_book", 
			"Total_Relationship_Count", 
			"Months_Inactive_12_mon", 
			"Contacts_Count_12_mon", 
			"Credit_Limit", 
			"Total_Revolving_Bal", 
			"Avg_Open_To_Buy", 
			"Total_Amt_Chng_Q4_Q1", 
			"Total_Trans_Amt", 
			"Total_Trans_Ct", 
			"Total_Ct_Chng_Q4_Q1", 
			"Avg_Utilization_Ratio"
		]
	
	edu_level_enc_order = [
		'Unknown', 
		'Uneducated',	
		'High School',
		'Graduate',
		'College',
		'Doctorate',
		'Post-Graduate'
	]

	card_cat_order = ["Blue", "Silver", "Gold", "Platinum"]

	income_cat_order = [
		'Unknown', 
		'Less than $40K',
		'$40K - $60K',
		'$60K - $80K',
		'$80K - $120K',
		'$120K +',
	]

	categories_order = [
		edu_level_enc_order, 
		income_cat_order, 
		card_cat_order
	]
	

	preprocessing_ct = ColumnTransformer(transformers=[
		("cat_onehot", OneHotEncoder(), cat_cols),
		("scale", StandardScaler(), num_cols),
		("ord_enc", OrdinalEncoder(categories=categories_order), ordinal_cols),
	],
	remainder="drop")

	return preprocessing_ct

"""
function that calculates the shapley values of two samples for class instance 1

x_test_enc: preprocessed x-test dataset
rf_model: Random forest model
id_instance_1: index of instance 1 in the x_test_enc dataset
id_instance_2: index of instance 2 in the x_test_enc dataset

returns: shapley values of the two samples
"""
def calc_shapley(x_test_enc, rf_model, id_instance_1, id_instance_2):
	class_instance = 1
	
	sample = x_test_enc
	masker = shap.maskers.Independent(data=x_test_enc)

	explainer = shap.Explainer(model=rf_model.predict_proba, # the function predict_proba
							masker=masker, seed=seed)

	shap_values = explainer(sample)

	shap_values_instance_1 = shap_values[id_instance_1][:, class_instance]
	shap_values_instance_2 = shap_values[id_instance_2][:, class_instance]

	return shap_values_instance_1.values, shap_values_instance_2.values

"""
Function that implements equation 3 and 4

ranking: The ranked indices that corresponds to the x_test_enc index of the sample
sample1_rank_index: index of sample1's ranking
sample2_rank_index: index of sample2's ranking
x_test_enc: Processed x_test data
rf_model: the random forest model

returns: abs_prod (i.e. the results of eq 4), sum (i.e. result of eq 3)
"""
def feature_contributions(ranking, sample1_rank_index, sample2_rank_index, x_test_enc, rf_model):
	id_instance_1 = int(ranking[sample1_rank_index])
	id_instance_2 = int(ranking[sample2_rank_index]) 

	shap_sample1, shap_sample2 = calc_shapley(x_test_enc, rf_model, id_instance_1, id_instance_2)

	shap_diff = np.subtract(shap_sample1, shap_sample2)
	val_diff = np.subtract(x_test_enc.iloc[id_instance_1], x_test_enc.iloc[id_instance_2])

	prod = shap_diff * val_diff

	sum = np.sum(prod)

	abs_prod = np.abs(prod) 

	return abs_prod, sum