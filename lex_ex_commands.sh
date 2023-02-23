#!/bin/bash

# python3 dlatkInterface.py -d shreyah -t sst -c message_id --message_field message --add_ngrams
# python3 dlatkInterface.py -d shreyah -t blog -c message_id --message_field message --add_ngrams
# python3 dlatkInterface.py -d shreyah -t emotions -c message_id --message_field message --add_ngrams
# python3 dlatkInterface.py -d shreyah -t polite -c message_id --message_field message --add_ngrams
# python3 dlatkInterface.py -d shreyah -t yelp -c message_id --message_field message --add_ngrams

# python3 dlatkInterface.py -d shreyah -t sst -c message_id -f 'feat$1gram$sst$message_id' --estimate_lda_topics --lda_lexicon_name sst --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 100 --num_topics 30 --save_lda_files lexicon_explanations/LDA/sst_lda_100_30 --lexicondb shreyah --language 'en'
# python3 dlatkInterface.py -d shreyah -t emotions -c message_id -f 'feat$1gram$emotions$message_id' --estimate_lda_topics --lda_lexicon_name emotions --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 100 --num_topics 30 --save_lda_files lexicon_explanations/LDA/emotions_lda_100_30 --lexicondb shreyah --language 'en'
# python3 dlatkInterface.py -d shreyah -t polite -c message_id -f 'feat$1gram$polite$message_id' --estimate_lda_topics --lda_lexicon_name polite --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 100 --num_topics 30 --save_lda_files lexicon_explanations/LDA/polite_lda_100_30 --lexicondb shreyah --language 'en'
# python3 dlatkInterface.py -d shreyah -t blog -c message_id -f 'feat$1gram$blog$message_id' --estimate_lda_topics --lda_lexicon_name blog --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 100 --num_topics 30 --save_lda_files lexicon_explanations/LDA/blog_lda_100_30 --lexicondb shreyah --language 'en'
# python3 dlatkInterface.py -d shreyah -t yelp -c message_id -f 'feat$1gram$yelp$message_id' --estimate_lda_topics --lda_lexicon_name yelp --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 100 --num_topics 30 --save_lda_files lexicon_explanations/LDA/yelp_lda_100_30 --lexicondb shreyah --language 'en'

# python3 ../dlatkInterface.py -d shreyah -t emotions -c message_id -f 'feat$1gram$emotions$message_id' --estimate_lda_topics --lda_lexicon_name emotions --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 50 --num_topics 30 --save_lda_files lexicon_explanations/LDA/emotions_lda_50_30 --lexicondb shreyah --language 'en'
python3 ../dlatkInterface.py -d shreyah -t blog -c message_id -f 'feat$1gram$blog$message_id' --estimate_lda_topics --lda_lexicon_name blog --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 50 --num_topics 30 --save_lda_files lexicon_explanations/LDA/blog_lda_50_30 --lexicondb shreyah --language 'en'
python3 ../dlatkInterface.py -d shreyah -t yelp -c message_id -f 'feat$1gram$yelp$message_id' --estimate_lda_topics --lda_lexicon_name yelp --mallet_path /home/sharath/mallet-2.0.8/bin/mallet --num_stopwords 50 --num_topics 30 --save_lda_files lexicon_explanations/LDA/yelp_lda_50_30 --lexicondb shreyah --language 'en'
