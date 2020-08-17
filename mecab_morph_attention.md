# mecab_morph_attention

데이터를 받고,  mecab pos_tagging에서 글자만 빼와서 하나의 문자로 만들었음

```python
def pos_sentence(pandas_name, column_name): 
  contents_list = pandas_name[column_name].tolist()                      #document만 리스트에 넣어두자
  preprocessed_docs = []
  for i in range(len(contents_list)):
    if type(contents_list[i]) != str:
      contents_list[i] = str(contents_list[i])
  preprocessed_contents = []
  for doc in contents_list:
    pos = mecab.pos(doc)
    preprocessed_contents.append(pos)
  all_words = []
  for sentence in preprocessed_contents:
    each_word = []
    for tokens in sentence:
      each_word.append(tokens[0])
    all_words.append(each_word)
  sentence_train = []
  for word in all_words:
    sentence_train.append(' '.join(word))
 

  return sentence_train

```

#### preprocess_sentence 함수를 통해 전처리(가-힣,a-z, A-Z, ".", "?", "!", ",") 빼고 다 담아주기

그다음 lambda를 통해 전처리 후 데이터들을 리스트로 만들어서 저장

```
## only for file kor.xlsx

def preprocess_sentence(sent):
    # 위에서 구현한 함수를 내부적으로 호출

    # 단어와 구두점 사이에 공백을 만듭니다.
    # Ex) "he is a boy." => "he is a boy ."
    sent = re.sub(r"([?.!,¿])", r" \1", sent)

    # (a-z, A-Z, ".", "?", "!", ",") 이들을 제외하고는 전부 공백으로 변환합니다.
    sent = re.sub(r"[^가-힣a-zA-Z!.?]+", r" ", sent)

    sent = re.sub(r"\s+", " ", sent)

    sent.strip()
    sent = '<start> ' + sent + ' <end>'

    return sent


def read_excel(df):
    df1 = df
    df1['kor'] = df['ko'].apply(lambda x : preprocess_sentence(x))
    df1['eng'] = df['en'].apply(lambda x : preprocess_sentence(x))

    # df['kor'] = df['kor'].apply(lambda x : x.split())

    # tar_input = df['eng'].apply(lambda x : "<sos>" + x)
    # tar_output= df['eng'].apply(lambda x : x + "<eos>")

    # sents_ko_in, sents_en_in, sents_en_out = [], [], []

    sents_ko_in = list(df['kor'])
    sents_en_in = list(df['eng'])
    # sents_en_out = list(tar_output)
    return sents_ko_in, sents_en_in
```

#### 토크나이즈 함수, 그리고 결과값 반환

```python
def tokenize(lang):
  lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
      filters='')
  lang_tokenizer.fit_on_texts(lang)

  tensor = lang_tokenizer.texts_to_sequences(lang)

  tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                         padding='post')

  return tensor, lang_tokenizer
  
input_tensor, inp_lang= tokenize(sents_ko_in)
target_tensor, targ_lang = tokenize(sents_en_in)
```

#### padding 후 max_length를 지정

```python
max_length_targ, max_length_inp = target_tensor.shape[1], input_tensor.shape[1] 

print(max_length_targ, max_length_inp)
```

20 32

#### 80-20 으로 train_test_split

```
input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor, target_tensor, test_size=0.2)


print(len(input_tensor_train), len(target_tensor_train), len(input_tensor_val), len(target_tensor_val))
```

60000 60000 15000 15000

#### 어떤식으로 돼있는지 확인

```
def convert(lang, tensor):
 for t in tensor:
  if t!=0:
   print ("%d ----> %s" % (t, lang.index_word[t]))
print ("Input Language; index to word mapping")
convert(inp_lang, input_tensor_train[0])
print ()
print ("Target Language; index to word mapping")
convert(targ_lang, target_tensor_train[0])
```

Input Language; index to word mapping 1 ----> <start> 

13486 ----> 바퀴벌레 

22 ----> 들  이런식으로 나오게 된다

#### 필요한 인수들 지정

```python
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_inp_size = len(inp_lang.word_index)+1
vocab_tar_size = len(targ_lang.word_index)+1

dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
```

#### Encoder class, LSTM을 활용하니 매우 결과가 안 좋게 나와 GRU

```python
class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    # Glorot 균등분포 초기값 설정기

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
```

#### attension

```python
class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # 시간순으로 계산되는 데이터를 브로드캐스팅하기위해
    # 이 변수를 지정
    query_with_time_axis = tf.expand_dims(query, 1)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis)+ self.W2(values)))
	# 나온 weight들을 tanh를 통해 score를 저장한다
    attention_weights = tf.nn.softmax(score, axis=1)
    # 각 항목별 score에서 softmax를 취해서 확률로 만들고
    context_vector = attention_weights * values
    # attension_weights와 value를 곱해서 최종값인 context_vector를 만든다
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

attention_layer = BahdanauAttention(10)
attention_result, attention_weights = attention_layer(sample_hidden, sample_output)

print("Attention result shape: (batch size, units) {}".format(attention_result.shape))
print("Attention weights shape: (batch_size, sequence_length, 1) {}".format(attention_weights.shape))
```

Attention result shape: (batch size, units) (64, 1024) Attention weights shape: (batch_size, sequence_length, 1) (64, 32, 1)

#### Decoder

```python
class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')# 균등분포 초기값
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for attention
    self.attention = BahdanauAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

sample_decoder_output, _, _ = decoder(tf.random.uniform((BATCH_SIZE, 1)),
                                      sample_hidden, sample_output)

print ('Decoder output shape: (batch_size, vocab size) {}'.format(sample_decoder_output.shape))
```

Decoder output shape: (batch_size, vocab size) (64, 19230)

#### complie / loss_function

```python
optimizer = tf.keras.optimizers.Adam()
# adam이 정말 최고로 좋습니다. 논문대로 하면 결과가 너무 느리고 안 좋게 나옴
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')

def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask # 마스킹

  return tf.reduce_mean(loss_)
```

checkpoint를 설정해주고, 훈련을 시키기 위한 함수

```python
@tf.function
def train_step(inp, targ, enc_hidden):
  loss = 0

  with tf.GradientTape() as tape:
    enc_output, enc_hidden = encoder(inp, enc_hidden)

    dec_hidden = enc_hidden

    dec_input = tf.expand_dims([targ_lang.word_index['<start>']] * BATCH_SIZE, 1)

    # Teacher forcing - feeding the target as the next input
    for t in range(1, targ.shape[1]):
      # passing enc_output to the decoder
      predictions, dec_hidden, _ = decoder(dec_input, dec_hidden, enc_output)

      loss += loss_function(targ[:, t], predictions)

      # using teacher forcing
      dec_input = tf.expand_dims(targ[:, t], 1)

  batch_loss = (loss / int(targ.shape[1]))

  variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, variables)

  optimizer.apply_gradients(zip(gradients, variables))

  return batch_loss
```

#### 훈련

```python
EPOCHS = 10

for epoch in range(EPOCHS):
  start = time.time()

  enc_hidden = encoder.initialize_hidden_state()
  total_loss = 0

  for (batch, (inp, targ)) in enumerate(dataset.take(steps_per_epoch)):
    batch_loss = train_step(inp, targ, enc_hidden)
    total_loss += batch_loss

    if batch % 100 == 0:
      print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                   batch,
                                                   batch_loss.numpy()))
  # saving (checkpoint) the model every 2 epochs
  if (epoch + 1) % 2 == 0:
    checkpoint.save(file_prefix = checkpoint_prefix)

  print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                      total_loss / steps_per_epoch))
  print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))
```

#### 평가

```python
def evaluate(sentence):
  attention_plot = np.zeros((max_length_targ, max_length_inp))

  sentence = preprocess_sentence(sentence)

  inputs = [inp_lang.word_index[i] for i in sentence.split(' ')]
  inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
                                                         maxlen=max_length_inp,
                                                         padding='post')
  inputs = tf.convert_to_tensor(inputs)

  result = ''

  hidden = [tf.zeros((1, units))]
  enc_out, enc_hidden = encoder(inputs, hidden)

  dec_hidden = enc_hidden
  dec_input = tf.expand_dims([targ_lang.word_index['<start>']], 0)

  for t in range(max_length_targ):
    predictions, dec_hidden, attention_weights = decoder(dec_input,
                                                         dec_hidden,
                                                         enc_out)

    # storing the attention weights to plot later on
    attention_weights = tf.reshape(attention_weights, (-1, ))
    attention_plot[t] = attention_weights.numpy()

    predicted_id = tf.argmax(predictions[0]).numpy()

    result += targ_lang.index_word[predicted_id] + ' '

    if targ_lang.index_word[predicted_id] == '<end>':
      return result, sentence, attention_plot

    # the predicted ID is fed back into the model
    dec_input = tf.expand_dims([predicted_id], 0)

  return result, sentence, attention_plot
```

#### 번역 함수

```python
def translate(sentence):
  result, sentence, attention_plot = evaluate(sentence)

  print('Input: %s' % (sentence))
  print('Predicted translation: {}'.format(result))
```

훈련한 체크포인트 불러들이고

```
# restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
```

#### 모든 문장이 이렇게 나온다(epoch=1)..

translate(u'선생 님 이 문장 이 이해 가 안 가 요 .')

Input: <start> 선생 님 이 문장 이 이해 가 안 가 요 . <end>

Predicted translation: i want to be a lot of the most popular . <end> 


