# Romantic-Relationship-Love-Data-Analysis-using-Python
This is an interesting data analytics research project of Facebook messenger data between two lovers being in a long-term committed relationship. The dataset contains data from July 2020 to February 2021. It required a huge data cleaning process on the JSON files after merging them into a single file. 

Md. Abrar Jahin, being the author of this project, tried to show the data analysis procedures using powerful Python libraries. He showed the process of how to code and left the rest upon the local machine to analyze the data. 

The project answered the quite interesting questions that were roaming around his head such as who texts more on average per day and per hour, who texts first, who replies late on an average, the sentiments throughout the day, the hour when they both remain free and love to talk much, and so on. 

### Some technical info for the data enthusiasts:
The code is available on GitHub. Feel free to fork, clone, play and ask anything about it.
VADER is a great NLP tool for analyzing text data sentiment in Python.
Word clouds were generated with Python’s WordCloud library and were decent (we conversed in traditional Bengali language using English letters).
Here are some links that I found useful and you might too!

**[Plotly](https://plot.ly/python/getting-started/)** —  an interactive plotting library for Python (and other languages) with detailed documentation pages.

**[VADER](http://t-redactyl.io/blog/2017/04/using-vader-to-handle-sentiment-analysis-with-social-media-text.html)** — this page gives a great walk-through of what’s behind the scenes and how to use the tool very practically. If you want to edit the lexicon and add your own terms with the `update` method:

```
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyser = SentimentIntensityAnalyzer()
new_words = {
    'greaaaat': 2.0,
    'shiiit': -2.0
}
analyser.lexicon.update(new_words)

```

**WordCloud** can be used easily after importing from wordcloud import WordCloud . Once that’s done, you can build the WC object and customize the colors, remove stop words and more.

Thanks to the sweet life partner of Abrar, who constantly supports him and appreciates every initiative he takes no matter how small or big a project is. A relationship requires proper direction and effort to be consistent and healthy. This analysis is nothing but a tremendous representation of evaluating the progress of a healthy relationship and taking measures to make it more efficient for both.

A thousand minutes of surfing in the ***Stack OverFlow***, ***Python Library Documentations*** resulted in a successful outcome. Have a happy life. Happy coding!!!

**To view the .ipynb Jupyter Notebook file, go to:**
https://nbviewer.jupyter.org/
and paste link: https://github.com/Abrar2652/Romantic-Relationship-Love-Data-Analysis-using-Python/blob/main/Facebook%20Romantic%20Relationship%20%26%20Love%20Data%20Analysis.ipynb

**Connect me on LinkedIn:**
https://www.linkedin.com/in/md-abrar-jahin-9a026018b/


**Add me on Facebook:**
https://www.facebook.com/in/abrar.jahin.2652
