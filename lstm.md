## 논문 모델 구현

### 전처리 

```python
def load_preprocessed_data(df):
    encoder_input, decoder_input, decoder_target = [], [], []
    for i in range(len(df)):
      src_line = df.iloc[i][0].strip()
      tar_line = df.iloc[i][1].strip()

      # source 데이터 전처리
      src_line_input = [w for w in preprocess_sentence(src_line).split()]

      # target 데이터 전처리
      tar_line = preprocess_sentence(tar_line)
      tar_line_input = [w for w in ("<sos> " + tar_line).split()]
      tar_line_target = [w for w in (tar_line + " <eos>").split()]

      encoder_input.append(src_line_input)
      decoder_input.append(tar_line_input)
      decoder_target.append(tar_line_target)

      # if i == num_samples - 1:
      #     break

    return encoder_input, decoder_input, decoder_target
```

- 훈련 과정에서 Teacher Forcing을 사용하므로, 훈련 시 사용할 decoder input 시퀀스와 실제 값에 해당하는 decoder_target을 분리하여 저장합니다. 

### 인코딩 및 패딩

```python
# 인코딩
tokenizer_kor = Tokenizer()
tokenizer_kor.fit_on_texts(sents_kor_in)
encoder_input_train = tokenizer_kor.texts_to_sequences(sents_kor_in)

tokenizer_eng = Tokenizer(filters="", lower=True)
tokenizer_eng.fit_on_texts(sents_eng_in)
tokenizer_eng.fit_on_texts(sents_eng_out)
decoder_input_train = tokenizer_eng.texts_to_sequences(sents_eng_in)
decoder_target_train = tokenizer_eng.texts_to_sequences(sents_eng_out)

# 패딩
encoder_input_train = pad_sequences(encoder_input_train, padding="post")
decoder_input_train = pad_sequences(decoder_input_train, padding="post")
decoder_target_train = pad_sequences(decoder_target_train, padding="post")
```

- 각각

이건 혹시 몰라서 추가합니다. 다들 아는 내용이라서 빼도 됩니다 ㅎㅎ

### 사용한 데이터 크기 

- 한국어 단어 집합의 크기 : 72499

- 영어 단어 집합의 크기 : 17412

- 한국어 Train input: (60000, 15)
-  영어 Train input: (60000, 19)
- 한국어 Test input: (15000, 14)
- 영어: Test input: (15000, 18)



## 번역기 만들기

#### 1. 문장 이상하게 출력되는 버전

- 인코더의 마지막 상태와 이전 디코더 상태를 initial state로 넣어줌
- Score는 60%까지 상승하나, 문장출력이 이상해짐 

```python
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
from tensorflow.keras.models import Model

latent_dim = 100

# 인코더 Layer 2개 
encoder_inputs = Input(shape=(None,))
enc_emb =  Embedding(src_vocab_size, latent_dim)(encoder_inputs) # 임베딩 층
enc_masking = Masking(mask_value=0.0)(enc_emb) # 패딩 0은 연산에서 제외

#MultiLayer LSTM
e_outputs, h1, c1 = LSTM(latent_dim, return_state=True, return_sequences=True)(enc_masking) # 은닉 상태와 셀 상태를 리턴 (output, hidden_state, cell_state)
encoder_lstm= LSTM(latent_dim, return_state=True, return_sequences=True) #모델에 넣어줌 
_, h2, c2 = encoder_lstm(e_outputs) # 은닉 상태와 셀 상태를 리턴
encoder_states = [h1, c1, h2, c2] # 인코더의 은닉 상태와 셀 상태를 저장

# 디코더 Layer 2개 
decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(tar_vocab_size, latent_dim) # 임베딩 층
dec_emb = dec_emb_layer(decoder_inputs)
dec_masking = Masking(mask_value=0.0)(dec_emb) # 패딩 0은 연산에서 제외

# 상태값 리턴을 위해 return_state는 True, 모든 시점에 대해서 단어를 예측하기 위해 return_sequences는 True
out_layer1 = LSTM(latent_dim, return_sequences=True, return_state=True)
d_outputs, dh1, dc1 = out_layer1(dec_masking,initial_state= [h2, c2]) # 인코더의 은닉 상태를 초기 은닉 상태(initial_state)로 사용
out_layer2 = LSTM(latent_dim, return_sequences=True, return_state=True)
final, dh2, dc2 = out_layer2(d_outputs, initial_state= [dh1, dc1])

# 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(final)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.summary()

```

원문 :  나는 오늘 자정에 한국으로 돌아 가요 .  

번역문 : i m going back to korea today at midnight .  

예측문 :  s s s s there there would there there   

원문 :  지금 잠을 자면 깨어나지 못할 거 같아서 지금 가요 .  

번역문 : if i fall asleep i might not get up so i will go right now .  

예측문 :  s s s s s there there would there   

원문 :  어제 밤에 왔고 오늘 밤에 가요 .  

번역문 : i came yesterday and i will leave today .  

예측문 :  s s s s s there there would there there   

원문 :  다음주 목요일 일에 한국으로 돌아 가요 .  

번역문 : i will be going back to korea next thursday .  

예측문 :  s s s s there there   

원문 :  그러나 인보이스의 단가는 잘못된 것 같아 .  

번역문 : but i think the price of invoice is wrong .  

예측문 :  s s s s there there there 



**Bleu score 0.05**



#### 2. 인코더의 각 상태를 Initial State로 넣어줌

- 문장 출력 잘 됨

- Val Acc 0.58

  ```python
  from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Masking
  from tensorflow.keras.models import Model
  
  latent_dim = 100
  
  # 인코더 Layer 2개 
  encoder_inputs = Input(shape=(None,))
  enc_emb =  Embedding(src_vocab_size, latent_dim)(encoder_inputs) # 임베딩 층
  enc_masking = Masking(mask_value=0.0)(enc_emb) # 패딩 0은 연산에서 제외
  
  #MultiLayer LSTM
  e_outputs, h1, c1 = LSTM(latent_dim, return_state=True, return_sequences=True)(enc_masking) # 은닉 상태와 셀 상태를 리턴 (output, hidden_state, cell_state)
  encoder_lstm= LSTM(latent_dim, return_state=True, return_sequences=True) #모델에 넣어줌 
  _, h2, c2 = encoder_lstm(e_outputs) # 은닉 상태와 셀 상태를 리턴
  encoder_states = [h1, c1, h2, c2] # 인코더의 은닉 상태와 셀 상태를 저장
  
  # 디코더 Layer 2개 
  decoder_inputs = Input(shape=(None,))
  dec_emb_layer = Embedding(tar_vocab_size, latent_dim) # 임베딩 층
  dec_emb = dec_emb_layer(decoder_inputs)
  dec_masking = Masking(mask_value=0.0)(dec_emb) # 패딩 0은 연산에서 제외
  
  # 상태값 리턴을 위해 return_state는 True, 모든 시점에 대해서 단어를 예측하기 위해 return_sequences는 True
  out_layer1 = LSTM(latent_dim, return_sequences=True, return_state=True)
  d_outputs, dh1, dc1 = out_layer1(dec_masking,initial_state= [h1, c1]) # 인코더의 은닉 상태를 초기 은닉 상태(initial_state)로 사용
  out_layer2 = LSTM(latent_dim, return_sequences=True, return_state=True)
  final, dh2, dc2 = out_layer2(d_outputs, initial_state= [h2, c2])
  
  # 모든 시점의 결과에 대해서 소프트맥스 함수를 사용한 출력층을 통해 단어 예측
  decoder_dense = Dense(num_decoder_tokens, activation='softmax')
  decoder_outputs = decoder_dense(final)
  
  model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
  
  model.summary()
  ```

  

## 번역기 실행

``` python
# 인코더 모델
encoder_model = Model(encoder_inputs, encoder_states)

# 디코더 모델 인풋 
decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_state_input_h1 = Input(shape=(latent_dim,))
decoder_state_input_c1 = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c, 
                         decoder_state_input_h1, decoder_state_input_c1]

# 임베딩 
dec_emb2= dec_emb_layer(decoder_inputs)

#디코더 모델
d_o, state_h, state_c = out_layer1(dec_emb2, initial_state=decoder_states_inputs[:2])
d_o, state_h1, state_c1 = out_layer2(d_o, initial_state=decoder_states_inputs[-2:])
decoder_states = [state_h, state_c, state_h1, state_c1]
decoder_outputs = decoder_dense(d_o)

decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

decoder_model.summary()
```

![image-20200817211352712](C:\Users\leeso\TIL\Project\Seq to Seq\model_summary\incoder_decoder_실행)



## 실행결과

원문 :  나는 오늘 자정에 한국으로 돌아 가요 .  

번역문 : i m going back to korea today at midnight .  

예측문 :  you come to korea .   



원문 :  지금 잠을 자면 깨어나지 못할 거 같아서 지금 가요 .  

번역문 : if i fall asleep i might not get up so i will go right now .  

예측문 :  you have to be a friend i m not good to me . 

 

원문 :  어제 밤에 왔고 오늘 밤에 가요 .  

번역문 : i came yesterday and i will leave today . 

 예측문 :  my friend i work .   



원문 :  다음주 목요일 일에 한국으로 돌아 가요 .  

번역문 : i will be going back to korea next thursday .  

예측문 :  you come to korea .   



원문 :  그러나 인보이스의 단가는 잘못된 것 같아 .  

번역문 : but i think the price of invoice is wrong .  

예측문 :  the most important thing . 



### BLEU

12000개 지금 돌리는 중. 

10개만 했을 때는 0.178

