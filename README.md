# Chinese Segmentation

# Requirements

## data

### Raw corpus

> people2014.tar.gz

### Processed dictionary

> people.dict.pkl

> python dict_utils.py

# Models

## Matching Based Methods

> python test_matching.py

## Hidden Markov Models

> python test_hmm.py

# Conditional Random Field

BiGRU + CRF
Epoch 37/100
2157/2157 [==============================] - 77s 36ms/step - loss: 4.7686 - accuracy: 0.9585 - val_loss: 5.4757 - val_accuracy: 0.9419

Model: "bi_rnncrf"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        multiple                  253184    
_________________________________________________________________
bidirectional (Bidirectional multiple                  18816     
_________________________________________________________________
dense (Dense)                multiple                  165       
_________________________________________________________________
crf (CRF)                    multiple                  25        
=================================================================
Total params: 272,190
Trainable params: 272,190
Non-trainable params: 0
_________________________________________________________________
None

Train done! Lasts: 3253.28s

# BiLSTM + Attention

Epoch 32/100
2157/2157 [==============================] - 68s 31ms/step - loss: 0.1030 - acc: 0.7230 - val_loss: 0.1610 - val_acc: 0.7040

Model: "bi_rnn_attention"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
embedding (Embedding)        multiple                  253184    
_________________________________________________________________
bidirectional (Bidirectional multiple                  18816     
_________________________________________________________________
attention (Attention)        multiple                  3168      
_________________________________________________________________
dense (Dense)                multiple                  165       
=================================================================
Total params: 275,333
Trainable params: 275,333
Non-trainable params: 0
_________________________________________________________________
None

Train done! Lasts: 2578.73s

# Performance

|model|precision|recall|f1_score|note|
|-----|---------|------|--------|----|
|bimm|0.9121|0.9536|0.9324||
|hmm|0.6921|0.6899|0.6910||
|jieba|0.8151|0.8092|0.8122||
|bilstm|0.8849|0.8701|0.8774||
|bigru+crf|0.8857|0.8760|0.8808|epochs=37|
|bilstm+attention|0.8439|0.7701|0.8053|epoch is about 10, not converge yet|
|bilstm+attention|0.8581|0.7850|0.8199|epochs=32|
|BERT|
|BERT+CRF|

