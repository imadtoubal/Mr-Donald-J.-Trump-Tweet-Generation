# Mr. Donald J. Trump Tweet Generation with LSTMs

* [![Open Word-Level In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imadtoubal/Mr-Donald-J.-Trump-Tweet-Generation/blob/master/word_level_text_generation.ipynb): Character level
* [![Open Character-Level In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/imadtoubal/Mr-Donald-J.-Trump-Tweet-Generation/blob/master/character_level_text_generation.ipynb): Word level


Text generation is a popular problem in natural language processing with machine learning. It ranges from simple email replies, word suggestions to simulating DNA sequences, and unfortunately fake news. This project aims to implement a generative model to learn the speech style of Mr. Donald Trump based on a dataset of his speeches; and then, automatically generate an unlimited amount of new speeches in the vein of Trump´s previous speeches.

# Dataset description
This project uses a [dataset from Kaggle that contained Donald J. Trump’s tweets](https://www.kaggle.com/davidg089/all-djtrum-tweets) from between May 2009 to August 2018. The dataset originally included 7 columns: `source`, `text`, `created_at`, `retweet_count`, `favorite_count`, `is_retweet`, and `id_str`. 

## Getting Started

This project was originally built on Google Colab. You can either upload the `.ipynb` notebooks to Colab or install dependencies locally on your computer.

### Using Google Colab

* Upload the `.ipynb` as a notebook in google colab. 
* Upload the dataset to the right folder.
* Change path to the dataset in the code.

### Run locally

You need to have Python installed in your computer.

1. Install `virtualenv`: 
    ```console
    pip install virtualenv
    ```
2. Create a Python virtual environment:
    ```console
    virtualenv venv
    ```
3. Activate virtual environment:
    1. Windows:
    ```console
    cd venv\Scripts
    activate
    cd ..\..
    ```
    2. Lunix / Mac:
    ```console
    source venv/bin/activate
    ```
4. Install libraries:
   
   ```console
   pip install -r requirements.txt
   ```
### Run the code

* Word level training code:
    ```console
    python word_level_text_generation.py
    ```
* Character level training code:
    ```console
    python character_level_text_generation.py
    ```

## Built With

* [Tensorflow (Keras)](https://www.tensorflow.org/) - The Machine Learning framework used

## Authors

* **Imad Eddine Toubal** - *Initial work* - [imadtoubal](https://github.com/imadtoubal)
* **How Lia** - *Initial work* 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details

## Acknowledgments

* Dataset by [David G. on Kaggle](https://www.kaggle.com/davidg089)
* Inspiration: [Generate text from Nietzsche's writings](https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py)

 Happy coding!
