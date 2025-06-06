import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.metrics import classification_report


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
def print_permutation_importances(perm_imp_train, perm_imp_test, cols, plot=False, save_plot=False, save_path="fig.png"):
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
ranking: a list of indices representing the ranked order of items in 'X_test'.
probs: the prediction probabilities of X_test
k: number of selected elements

returns: a dataframe with the top k elements, their true label and their predicted label
"""
def get_topK(X_test, Y_test, y_pred, ranking, probs, k):
	#sorts according to ranking
	x_test_ranked = X_test.iloc[ranking]
	y_test_ranked = Y_test.iloc[ranking]
	probs_ranked = probs[ranking]
	y_pred_ranked = y_pred[ranking]

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
path: to save figures for permutation importance plots

returns: rankings and the prediction probabilities
"""
def fit_and_rank(model, X_train_enc, Y_train, X_test_enc, Y_test, path):
	# Fit to new data
	model.fit(X_train_enc, Y_train)
	Y_pred = model.predict(X_test_enc)
		
	# Check accuracy
	print_accuracy(model, X_train_enc, Y_train, X_test_enc, Y_test, Y_pred)

	# compute and plot feature importances
	train_perm_imp = permutation_importance(model, X_train_enc, Y_train, n_repeats=10, random_state=seed)
	test_perm_imp = permutation_importance(model, X_test_enc, Y_test, n_repeats=10, random_state=seed)
	print_permutation_importances(train_perm_imp, test_perm_imp, X_train_enc.columns, plot=True, save_plot=True, save_path=path)
	
	#Ranking
	probs = model.predict_proba(X_test_enc)[:, 1]
	ranking = probs.argsort()[::-1]

	return ranking, probs, Y_pred
	

def attrition_flag_map(df):
	return df['Attrition_Flag'].map({'Existing Customer': 0, 'Attrited Customer': 1})

def preprocessing_pipe():	
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
	

	preprocessing_pipe = ColumnTransformer(transformers=[
		("cat_onehot", OneHotEncoder(), cat_cols),
		("scale", StandardScaler(), num_cols),
		("ord_enc", OrdinalEncoder(categories=categories_order), ordinal_cols),
	],
	remainder="drop")

	return preprocessing_pipe