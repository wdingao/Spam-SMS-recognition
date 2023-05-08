import re
import jieba
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, DataLoader, random_split
from transformers import BertModel, BertTokenizer

'''
这是一个数据预处理程序
txt文件的内容为label  sms
label为短信是否为垃圾短信的判定，0为正常短信，1为垃圾短信。
sms为短信内容
这个程序将文本处理后向量化返回
'''


def read_dataset(data_set_path):
    # 传入数据文件路径，读取文件
    with open(data_set_path, 'r', encoding='utf-8') as f:
        data = f.readlines()

    # 分离标签
    labels_texts = [(int(line.split('\t')[0]), line.split('\t')[1].strip()) for line in data]

    # 返回值格式[(标签,[短信内容])]
    return labels_texts


def data_clean(data):
    # 对短信进行数据清洗，仅仅保留中文元素
    # data 格式[(标签,[短信内容])]
    pattern = re.compile(r'[^\u4e00-\u9fff]')
    for i in range(len(data)):
        text = data[i][1]
        data[i] = (data[i][0], pattern.sub(' ', text))

    # 返回值格式[(标签,[短信内容（已清洗）])]
    return data


def sms_tokenizer(data):
    # 分词器对信息进行分词，并去除所有的空格
    # data 格式[(标签,[短信内容])]
    for i in range(len(data)):
        text = data[i][1]
        text = jieba.lcut(text)
        text = [word for word in text if word != ' ']
        data[i] = (data[i][0], text)

    # 返回值格式[(标签,[短信内容(已分词)])]
    return data


def del_stopwords(data):
    # 去除停用词
    # data 格式[(标签,[短信内容])]
    stopwords = [line.strip() for line in open('data/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
    for i in range(len(data)):
        text = data[i][1]
        text = [word for word in text if word not in stopwords]
        data[i] = (data[i][0], text)

    # 返回值格式[(标签,[短信内容(已过滤停用词)])]
    return data


def get_word_vector(data):
    # 获取词向量
    # 传入的data 格式[(标签,[短信内容])]

    # 加载模型，初始化BertTokenizer，并迁移至GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertModel.from_pretrained('bert-base-chinese').to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    for i in range(len(data)):
        # 将短信转换为BERT输入，即在短信内容前加上'[CLS]',后面加上'[SEP]',之后获取
        text = data[i][1]
        text = ['[CLS]'] + text + ['[SEP]']
        # 将分词转换为Token ID
        token_ids = tokenizer.convert_tokens_to_ids(text)

        # 将token ID序列转换为PyTorch张量
        tokens_tensor = torch.tensor([token_ids]).to(device)

        # 使用BERT模型计算词向量
        with torch.no_grad():
            outputs = model(tokens_tensor)
            encoded_layers = outputs[0]
        token_vectors = encoded_layers[0]

        # 保存词向量
        data[i] = (data[i][0], token_vectors)

    # 返回值格式[(标签，tensor([[]])]
    return data


def data_packing(data):
    # 对处理后的数据进行重新封装

    # 分离标签和数据
    label_list = [row[0] for row in data]
    data_list = [row[1] for row in data]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 对文本张量进行填充，使得其大小统一
    data_list = pad_sequence(data_list, batch_first=True, padding_value=0).to(device)
    data_tensor = data_list

    # 获取标签的张量并转换维度
    label_tensor = torch.tensor(label_list, dtype=torch.long).to(device)
    label_tensor = label_tensor.unsqueeze(1)

    # 创建数据集
    dataset = TensorDataset(data_tensor, label_tensor)

    # 划分训练集、测试集、验证集，比例为8：1：1
    dataset_len = len(dataset)

    # 计算训练集、验证集和测试集长度
    train_len = int(0.8 * dataset_len)
    val_len = int(0.1 * dataset_len)
    test_len = dataset_len - train_len - val_len

    # 划分数据集
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_len, val_len, test_len])

    # 创建数据加载器
    batch_size = 64
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_data_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader


def get_sequence(sms):
    # 模型训练完成后，该函数获取输入短信的张量用以使用模型进行推理，基本就是把上面的流程走了一遍。因此不做过多注释。早知道前面函数不用data的list了
    stopwords = [line.strip() for line in open('data/cn_stopwords.txt', 'r', encoding='utf-8').readlines()]
    sms = re.compile(r'[^\u4e00-\u9fff]').sub(' ', sms)
    sms = jieba.lcut(sms)
    sms = [word for word in sms if (word != ' ' and word not in stopwords)]
    model = BertModel.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    sms = ['[CLS]'] + sms + ['[SEP]']
    token_ids = tokenizer.convert_tokens_to_ids(sms)
    tokens_tensor = torch.tensor([token_ids])
    with torch.no_grad():
        outputs = model(tokens_tensor)
        encoded_layers = outputs[0]
    token_vectors = encoded_layers[0]
    # 对文本张量进行填充，使得其大小统一
    sms_vector = torch.tensor(token_vectors).unsqueeze(0)
    return sms_vector


def data_process(data_set_path):
    # 处理数据
    print('读取数据')
    data = read_dataset(data_set_path)
    print('读取完成\n清洗数据')
    data = data_clean(data)
    print('清洗完成\n分词')
    data = sms_tokenizer(data)
    print('分词完成\n过滤停用词')
    data = del_stopwords(data)
    print('停用词过滤完成\n特征提取')
    data = get_word_vector(data)
    print('特征提取完成\n准备数据集')
    # 接下来，对获取的数据进行封装
    print(data)
    train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader = data_packing(data)
    print("数据集准备完成")
    return train_dataset, val_dataset, test_dataset, train_data_loader, val_data_loader, test_data_loader
