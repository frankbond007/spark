import os
import subprocess
from pyspark.sql import SparkSession
from pyspark import SparkFiles, SparkConf, SparkContext
from pyspark.sql.functions import udf, when, col, concat_ws
from pyspark.sql.types import StringType, ArrayType, FloatType, IntegerType
from pyspark.ml.feature import CountVectorizer, StopWordsRemover, Tokenizer
from pyspark.ml.clustering import LDA
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from nltk import download
import tarfile
download('words')
download('vader_lexicon')
download('punkt')
download('stopwords')
download('wordnet')
download('averaged_perceptron_tagger')

cmd = ("tar -czvf nltk_data.tar.gz corpora/words corpora/stopwords corpora/vader_lexicon copora/wordnet corpora/averaged_perceptron_tagger copora/punkt copora/averaged_perceptron_tagger")
result = subprocess.run(cmd.split(), capture_output=True, text=True)
spark_context = SparkContext(conf=SparkConf())
spark_context.addFile("nltk_data.tar.gz")
spark = SparkSession.builder \
    .appName("SentimentAnalysis") \
    .getOrCreate()


def setup_nltk_environment(_):
    """
    Function to be executed on each worker node.
    """
    # Extract nltk_data.tar.gz in the current working directory
    with tarfile.open(SparkFiles.get("nltk_data.tar.gz"), "r:gz") as tar:
        tar.extractall()

    # Set NLTK_DATA environment variable
    os.environ['NLTK_DATA'] = os.path.join(os.getcwd(), 'nltk_data')
    return [1]
# Dummy Spark job to trigger the execution on each node
spark_context.parallelize(range(spark_context.defaultParallelism), spark_context.defaultParallelism).mapPartitions(setup_nltk_environment).collect()

english_vocab = set(w.lower() for w in words.words())
sia = SentimentIntensityAnalyzer()
lemmatizer = WordNetLemmatizer()
@udf(StringType())
def extract_opinions(text):
    tokenized = word_tokenize(text)
    opinions = [word for word, tag in pos_tag(tokenized) if tag in ['JJ', 'JJR', 'JJS']]
    return ', '.join(opinions)

@udf(StringType())
def get_sentiment(text):
    simple_responses = ['no', 'yes']
    if text.strip().lower() in simple_responses:
        return 'neutral'
    score = sia.polarity_scores(text)['compound']
    return 'positive' if score > 0.05 else 'negative' if score < -0.05 else 'neutral'

@udf(StringType())
def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    return ' '.join([lemmatizer.lemmatize(token) for token in tokens if token.isalpha()])

@udf(IntegerType())
def get_dominant_topic(vector):
    return int(vector.argmax())

df = spark.read.option("header", "true").csv("input.csv")

filtered_df = df.filter(
    (col('QuestionType') == 'User Comment') &
    col('ParticipantResponse').isNotNull()
)
filtered_df.show()
filtered_df = filtered_df.filter(col('QuestionType') == 'User Comment')
filtered_df = filtered_df.withColumn("Sentiment", get_sentiment(col("ParticipantResponse")))
filtered_df = filtered_df.withColumn("Opinions", extract_opinions(col("ParticipantResponse")))
filtered_df = filtered_df.withColumn("ProcessedText", preprocess_text(col("ParticipantResponse")))
filtered_df.show()
tokenizer = Tokenizer(inputCol="ProcessedText", outputCol="ProcessedText_tokens")
filtered_df = tokenizer.transform(filtered_df)

remover = StopWordsRemover(inputCol="ProcessedText_tokens", outputCol="ProcessedText_clean")
filtered_df = remover.transform(filtered_df)

vectorizer = CountVectorizer(inputCol="ProcessedText_clean", outputCol="features")
model = vectorizer.fit(filtered_df)
filtered_df = model.transform(filtered_df)

lda = LDA(k=5, maxIter=10)
lda_model = lda.fit(filtered_df)
transformed_df = lda_model.transform(filtered_df)

filtered_df = transformed_df.withColumn("DominantTopic", get_dominant_topic(col("topicDistribution")))

topic_indices = lda_model.describeTopics(10)
vocab_list = model.vocabulary

def map_termID_to_Word(termIndices):
    return [vocab_list[termID] for termID in termIndices]

term_to_word_udf = udf(map_termID_to_Word, ArrayType(StringType()))
topics_df = topic_indices.withColumn("TopicKeywords", term_to_word_udf(col("termIndices")))
print("topicsdf")
topics_df.show()
#final_df = filtered_df.join(topics_df, on="DominantTopic", how="left")
topics_df = topics_df.withColumn("termIndices", concat_ws(",", col("termIndices")))
for column in filtered_df.columns:
    filtered_df = filtered_df.withColumn(column, col(column).cast("string"))
for column in topics_df.columns:
    topics_df = topics_df.withColumn(column, col(column).cast("string"))
filtered_df.write.option("header", "true").csv("filtered_df.csv")
#final_df.write.option("header", "true").csv("finl_df.csv")
topics_df.write.option("header", "true").csv("topics.csv")
spark.stop()