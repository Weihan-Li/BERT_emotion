from transformers import BertTokenizer, BertConfig, TFBertModel
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback

# 你的文件的路径
config_path = 'chinese-pert-large/bert_config.json'
vocab_path = 'chinese-pert-large/vocab.txt'
model_path = 'chinese-pert-large/bert_model.ckpt'

# 加载tokenizer
tokenizer = BertTokenizer.from_pretrained(vocab_path, do_lower_case=False)

# 加载BERT配置
config = BertConfig.from_json_file(config_path)

# 创建一个新的BERT模型实例
model = TFBertModel(config)

# 从checkpoint文件加载权重
ckpt_reader = tf.train.load_checkpoint(model_path)

bert_prefix = 'bert'

# 将权重赋值给模型
for weight in model.trainable_weights:
    # 这一步是将模型的权重名字和checkpoint的权重名字对应起来
    # 因为在训练时可能使用了不同的框架，所以名字可能会有所不同
    name = weight.name.split(':')[0]
    name = bert_prefix + '/' + name
    # 获取checkpoint中的权重
    ckpt_weight = ckpt_reader.get_tensor(name)
    # 如果权重的shape不匹配，那么可能需要reshape一下
    if ckpt_weight.shape != weight.shape:
        ckpt_weight = np.reshape(ckpt_weight, weight.shape)
    # 将checkpoint的权重赋值给模型
    weight.assign(ckpt_weight)

print("Model weights have been loaded successfully.")

# 加载数据
df_train = pd.read_json('usual_train.txt')
df_eval = pd.read_json('usual_eval_labeled.txt')
df_test = pd.read_json('usual_test_labeled.txt')

# 使用LabelEncoder对标签进行编码
encoder = LabelEncoder()
df_train['label'] = encoder.fit_transform(df_train['label'])
df_eval['label'] = encoder.transform(df_eval['label'])
df_test['label'] = encoder.transform(df_test['label'])

# 进行分词处理并转化为BERT可识别的输入格式
def prep_data(text, max_length):
    return tokenizer.encode_plus(text,
                                 add_special_tokens=True,
                                 max_length=max_length,
                                 pad_to_max_length=True,
                                 return_attention_mask=True,
                                 return_token_type_ids=False,
                                 return_tensors='tf')

max_length = 32

def prepare_set(dataset, max_length):
    input_ids = []
    attention_masks = []

    for text in dataset['content']:
        encoded_dict = prep_data(text, max_length)
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = tf.concat(input_ids, axis=0)
    attention_masks = tf.concat(attention_masks, axis=0)

    return {'input_ids': input_ids, 'attention_mask': attention_masks}

X_train = prepare_set(df_train, max_length)
y_train = df_train['label'].values
X_eval = prepare_set(df_eval, max_length)
y_eval = df_eval['label'].values
X_test = prepare_set(df_test, max_length)
y_test = df_test['label'].values


# 添加一个全连接层进行情绪分类
input_ids = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
attention_mask = tf.keras.layers.Input(shape=(max_length,), dtype=tf.int32)
outputs = model([input_ids, attention_mask])[0]
outputs = outputs[:, 0, :]
outputs = Dense(units=len(encoder.classes_), activation='softmax')(outputs)

# 构建模型
model = Model(inputs=[input_ids, attention_mask], outputs=outputs)
model.compile(optimizer=Adam(lr=1e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
batch_size = 64  # 替换为你实际使用的批量大小
# 训练模型，并添加回调函数
print_callback = LambdaCallback(
    on_batch_end=lambda batch, logs: print(
        "\nBatch {} finished. Loss: {}. Accuracy: {}.".format(
            batch, logs['loss'], logs['accuracy']
        )
    ),
    on_epoch_end=lambda epoch, logs: print(
        "Epoch {} finished. Val Loss: {}. Val Accuracy: {}.".format(
            epoch, logs['val_loss'], logs['val_accuracy']
        )
    )
)
# 自定义回调函数
def save_model_callback(epoch, logs):
    if (epoch + 1) % 12 == 0:  # 每隔12个epoch保存一次模型
        filepath = 'BERT_{epoch:02d}.h5'.format(epoch=epoch + 1)
        model.save(filepath)
        print("Saved model at epoch", epoch + 1)

# 创建ModelCheckpoint回调函数，用于仅在保存时显示相关信息
checkpoint_callback = ModelCheckpoint(
    filepath='temp.h5',
    save_weights_only=False,
    verbose=0
)

# 创建LambdaCallback回调函数，调用自定义的保存模型回调函数
save_callback = LambdaCallback(on_epoch_end=save_model_callback)

# 训练模型，并添加回调函数
model.fit([X_train['input_ids'], X_train['attention_mask']], y_train, epochs=24, validation_data=([X_eval['input_ids'], X_eval['attention_mask']], y_eval), callbacks=[print_callback, save_callback])
# 使用测试数据集进行评估
model.evaluate([X_test['input_ids'], X_test['attention_mask']], y_test, callbacks=[print_callback])