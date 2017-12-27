---
layout: post
post: How to plot a Confusion Matrix in Python
---

Here is an example to show how you can plot the Confusion Matrix from Scikit-Learn to display in a more intuitive visual format. 

The documentation for **[Confusion Matrix](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html)** is pretty good, but I struggled to find a quick way to add labels and visualize the output into a 2x2 table.

For a good introductory read on confusion matrix check out this great post:  

<http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology>

Let's go through a quick Logistic Regression example using Scikit-Learn where in the end we will visualize the confusion matrix using matplotlib:

{% highlight python %}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
data = datasets.load_iris()

df = pd.DataFrame(data.data, columns=data.feature_names)

{% endhighlight %}

We can examine our data quickly using Pandas correlation function to pick a suitable feature for our logistic regression.


{% highlight python %}
corr = df.corr()
print(corr.Target)
{% endhighlight %}



{% highlight text %}
    >>> output
    sepal length (cm)    0.782561
    sepal width (cm)    -0.419446
    petal length (cm)    0.949043
    petal width (cm)     0.956464
    Target               1.000000
    Name: Target, dtype: float64
{% endhighlight %}

So, let's pick Petal Width (cm) as our (X) independent variable. For our Target/dependent variable (Y) we can pick the Setosa class. The Target class actually has three choices, to simplify our task and narrow it down to a binary classifier we will pick Setosa (0 or 1): either it is Setosa (1) or it is Not Setosa (0).

{% highlight python %}
print(data.target_names)
{% endhighlight %}

{% highlight text %}
    >>> output
    array(['setosa', 'versicolor', 'virginica'],
      dtype='<U10')
{% endhighlight  %}  

Let's now create X and Y:

{% highlight python %}
x = df.iloc[0: ,3].reshape(-1,1)
y = (data.target == 0).astype(np.int)
{% endhighlight  %}

We will split our data into a test and train sets, then start building our Logistic Regression model.

{% highlight python %}
from sklearn.cross_validation import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 0)

from sklearn.linear_model import LogisticRegression
logit = LogisticRegression(random_state= 0)
logit.fit(x_train, y_train)

y_predicted = logit.predict(x_test)
{% endhighlight %}

Now, let's examine our confusion matrix:

{% highlight python %}
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_predicted)
print(cm)
{% endhighlight %}

{% highlight text %}
    >>> output
    [[19  0]
    [ 0 11]]
{% endhighlight %}

The confusion matrix tells us we got everything classified correctly (in terms of: Setosa, or Not Setosa). A better way to visualize this can be accomplished with the code below:

{% highlight python %}
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Wistia)
classNames = ['Negative','Positive']
plt.title('Setosa or Not Setosa Confusion Matrix - Test Data')
plt.ylabel('True label')
plt.xlabel('Predicted label')
tick_marks = np.arange(len(classNames))
plt.xticks(tick_marks, classNames, rotation=45)
plt.yticks(tick_marks, classNames)
s = [['TN','FP'], ['FN', 'TP']]

for i in range(2):
    for j in range(2):
        plt.text(j,i, str(s[i][j])+" = "+str(cm[i][j]))
plt.show()
{% endhighlight %}

![png]({{ site.baseurl }}/images/cm.png =350)

To plot and display the decision boundary that separates the two classes (Setosa or Not Setosa ):

{% highlight python %}
from matplotlib.colors import ListedColormap
X_set, y_set = x_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, logit.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Train set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
{% endhighlight %}

![png]({{ site.baseurl }}/images/logit.png =350)

Hope this helps.