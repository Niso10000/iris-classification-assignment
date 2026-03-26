# דוח פרויקט סיווג Iris

## מבוא

פרויקט זה מבצע סיווג של פרחי Iris לשלוש קטגוריות באמצעות ארבעה מודלים של למידת מכונה: KNN, SVM, Random Forest ו-Logistic Regression. המטרה היא להשוות את הביצועים של המודלים על מערך נתונים קלאסי, ולבחור את המודל הטוב ביותר על בסיס מדדי דיוק, precision, recall ו-F1.

## 1. תיאור מערך הנתונים

מערך הנתונים Iris הוא מערך נתונים קלאסי בלמידת מכונה, המכיל 150 דוגמאות של פרחי Iris. כל דוגמה כוללת ארבע תכונות מספריות:

- אורך ספטל (sepal length) בס"מ
- רוחב ספטל (sepal width) בס"מ
- אורך פטל (petal length) בס"מ
- רוחב פטל (petal width) בס"מ

הקטגוריות הן שלושה סוגי פרחים:
- Setosa (50 דוגמאות)
- Versicolor (50 דוגמאות)
- Virginica (50 דוגמאות)

הנתונים מחולקים ל-80% אימון ו-20% בדיקה, עם שמירה על איזון הקטגוריות (stratified split).

## 2. עיבוד ראשוני

- טעינת הנתונים מ-CSV.
- קידוד תוויות לקטגוריות מספריות באמצעות LabelEncoder.
- חלוקת הנתונים ל-train/test (80/20).
- סטנדרטיזציה של התכונות באמצעות StandardScaler, כדי להבטיח שכל התכונות יהיו באותו סולם.

## 3. מודלים ומבנה

המודלים שנבדקו:
- **KNN (K-Nearest Neighbors)**: מודל לא פרמטרי המבוסס על מרחקים בין דוגמאות.
- **SVM (Support Vector Machine)**: מודל המחפש היפר-מישור המפריד בין הקטגוריות.
- **Random Forest**: מודל ensemble של עצי החלטה, המפחית overfit.
- **Logistic Regression**: מודל ליניארי לבעיות סיווג רב-קטגוריות.

כל מודל אומן על הנתונים המעובדים, והוערך באמצעות cross-validation של 5 קפלים.

## 4. מתודולוגיית הערכה

- מדדי הערכה: accuracy, precision (macro), recall (macro), F1 (macro).
- cross-validation: 5-fold stratified.
- הערכה על קבוצת הבדיקה (test set).

## 5. תוצאות נומריות

| מודל               | Accuracy | Precision | Recall | F1     |
|--------------------|----------|-----------|--------|--------|
| KNN                | 0.9333   | 0.9444    | 0.9333 | 0.9327 |
| SVM                | 0.9667   | 0.9697    | 0.9667 | 0.9666 |
| Random Forest      | 0.9000   | 0.9024    | 0.9000 | 0.8997 |
| Logistic Regression| 0.9333   | 0.9333    | 0.9333 | 0.9333 |

## 6. מטריצות בלבול

![Confusion Matrix KNN](../outputs/confusion_matrix_KNN.png)
![Confusion Matrix SVM](../outputs/confusion_matrix_SVM.png)
![Confusion Matrix Random Forest](../outputs/confusion_matrix_RandomForest.png)
![Confusion Matrix Logistic Regression](../outputs/confusion_matrix_LogisticRegression.png)

## 7. ניתוח גרפים

- **גרף השוואת מודלים**: מראה את הביצועים של כל מודל בכל מדד. SVM בולט בדיוק הגבוה ביותר.
- **חשיבות תכונות**: תכונות הפטל (petal length ו-petal width) הן החשובות ביותר, כפי שנראה בגרף.
- **מפת קורלציה**: קורלציה גבוהה בין תכונות הפטל, מה שמצביע על multicollinearity אפשרית.
- **גבול החלטה**: מראה כיצד המודל מפריד בין הקטגוריות בשני מימדים.
- **cross-validation box plot**: מראה את היציבות של כל מודל על פני הקפלים.
- **עקומת למידה**: מראה את השיפור בביצועים עם יותר נתוני אימון.

![Model Comparison](../outputs/model_comparison.png)
![Feature Importance](../outputs/feature_importance.png)
![Correlation Heatmap](../outputs/correlation_heatmap.png)
![Decision Boundary](../outputs/decision_boundary.png)
![Cross Validation](../outputs/cross_validation.png)
![Learning Curve](../outputs/loss_curve.png)

## 8. ניתוח תוצאות

SVM הראה את הביצועים הטובים ביותר עם accuracy של 96.67%, מה שמצביע על יכולתו הטובה להפריד בין הקטגוריות. KNN ו-Logistic Regression היו דומים עם 93.33%, בעוד Random Forest היה הנמוך ביותר עם 90%. הסיבה לכך עשויה להיות שהנתונים הם ליניאריים ברובם, ו-SVM מתמודד טוב עם זה. Random Forest עלול להיות overfit בגלל מספר העצים.

מה עבד: כל המודלים הגיעו לדיוק מעל 90%, מה שמראה שהנתונים קלים לסיווג. מה לא: Random Forest לא בלט, אולי בגלל hyperparameter tuning חסר.

## 9. מסקנות והמלצה

המודל המומלץ הוא SVM, בגלל הדיוק הגבוה והיציבות. בעתיד, ניתן לבדוק hyperparameter tuning או מודלים נוספים כמו neural networks.

## 10. הצעת המשך

- ניסוי עם hyperparameter search (GridSearchCV).
- הרחבת מאפיינים או שימוש ב-feature selection.
- השוואה עם מודלים מתקדמים יותר.


## 8. מסקנות

- מה עבד, מה לא
- המלצה על המודל הטוב ביותר

## 9. הצעת המשך

- ניסוי hyperparameter search
- הרחבת מאפיינים
