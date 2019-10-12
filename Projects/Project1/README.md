# CoE 197-Z Deep Learning Project
### [Kannada MNIST](https://www.kaggle.com/c/Kannada-MNIST)
##### MNIST like datatset for Kannada handwritten digits

## Bored of MNIST?
The goal of this competition is to provide a simple extension to the classic [MNIST competition](https://www.kaggle.com/c/digit-recognizer/) we're all familiar with. Instead of using Arabic numerals, it uses a recently-released dataset of Kannada digits.

Kannada is a language spoken predominantly by people of Karnataka in southwestern India. The language has roughly 45 million native speakers and is written using the Kannada script. [Wikipedia](https://en.wikipedia.org/wiki/Kannada)

![Kannada](https://storage.googleapis.com/kaggle-media/competitions/Kannada-MNIST/kannada.png)

This competition uses the same format as the [MNIST competition](https://www.kaggle.com/c/digit-recognizer/) in terms of how the data is structured, but it's different in that it is a synchronous re-run Kernels competition. You write your code in a Kaggle Notebook, and when you submit the results, your code is scored on both the public test set, as well as a private (unseen) test set.

## Technical Information
All details of the dataset curation has been captured in the paper titled: Prabhu, Vinay Uday. "Kannada-MNIST: A new handwritten digits dataset for the Kannada language." arXiv preprint [arXiv:1908.01242 (2019)](https://arxiv.org/abs/1908.01242)

The github repo of the author [can be found here](https://github.com/vinayprabhu/Kannada_MNIST).

On the [originally-posted dataset](https://www.kaggle.com/higgstachyon/kannada-mnist), the author suggests some interesting questions you may be interested in exploring. Please note, although this dataset has been released in full, the purpose of this competition is for practice, not to find the labels to submit a perfect score.

In addition to the main dataset, the author also disseminated an additional real world handwritten dataset (with 10k images), termed as the 'Dig-MNIST dataset' that can serve as an out-of-domain test dataset. It was created with the help of volunteers that were non-native users of the language, authored on a smaller sheet and scanned with different scanner settings compared to the main dataset. This 'dig-MNIST' dataset serves as a more difficult test-set (An accuracy pf 76.1% was reported in the paper cited above) and achieving approximately 98+% accuracy on this test dataset would be rather commendable.

## Acknowledgments
Kaggle thanks [Vinay Prabhu](https://www.kaggle.com/higgstachyon) for providing this interesting dataset for a Playground competition.

Image reference: https://www.researchgate.net/figure/speech-for-Kannada-numbers_fig2_313113588

## Data Description
The data files `train.csv` and `test.csv` contain gray-scale images of hand-drawn digits, from zero through nine, in the Kannada script.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.

The training data set, `train.csv`, has 785 columns. The first column, called `label`, is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.

Each pixel column in the training set has a name like `pixel{x}`, where `x` is an integer between 0 and 783, inclusive. To locate this pixel on the image, suppose that we have decomposed `x` as `x = i * 28 + j`, where `i` and `j` are integers between 0 and 27, inclusive. Then `pixel{x}` is located on row `i` and column `j` of a 28 x 28 matrix, (indexing by zero).

For example, `pixel31` indicates the pixel that is in the fourth column from the left, and the second row from the top, as in the ascii-diagram below.

Visually, if we omit the "pixel" prefix, the pixels make up the image like this:

`000 001 002 003 ... 026 027`  
`028 029 030 031 ... 054 055`  
`056 057 058 059 ... 082 083`  
` |   |   |   |  ...  |   |`  
`728 729 730 731 ... 754 755`  
`756 757 758 759 ... 782 783`   

The test data set, `test.csv`, is the same as the training set, except that it does not contain the `label` column.

The evaluation metric for this contest is the categorization accuracy, or the proportion of test images that are correctly classified. For example, a categorization accuracy of 0.97 indicates that you have correctly classified all but 3% of the images.

## Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format
- **Dig-MNIST.csv** - an additional labeled set of characters that can be used to validate or test model results before submitting to the leaderboard
