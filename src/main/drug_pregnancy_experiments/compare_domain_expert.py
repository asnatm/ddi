import os

import pandas as pd
from sklearn.metrics import roc_curve, auc

os.chdir('..\\..\\..')

final_class = pd.read_excel('output\\data\\covid_drugs_predictions from maya.xlsx')
final_class  = final_class[["Risk Probability Predicted by Model","Classifiaction by TIS Zerifin"]]
final_class = final_class[~final_class["Classifiaction by TIS Zerifin"].isna()]
final_class = final_class[~final_class["Classifiaction by TIS Zerifin"].str.contains('No Data')]

final_class["Classifiaction by TIS Zerifin"] = final_class["Classifiaction by TIS Zerifin"].str.replace('*','')


fpr, tpr, _ = roc_curve(final_class["Classifiaction by TIS Zerifin"], final_class['Risk Probability Predicted by Model'],pos_label="Limited")
roc_auc = auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig("output\\SHAP\\auc_expert.pdf",bbox_inches='tight',pad_inches =0.2)
plt.show()


for x in range(100):
	import sklearn
	t=x/100#best t=0.4
	kappa= sklearn.metrics.cohen_kappa_score(final_class["Classifiaction by TIS Zerifin"], ["Safe" if x <t else "Limited" for x in final_class['Risk Probability Predicted by Model']  ])
	print(t, kappa)