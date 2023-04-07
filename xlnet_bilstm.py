#from pytorch_transformers import XLNetTokenizer,XLNetModel,XLNetForSequenceClassification
from pytorch_transformers import AdamW
import torch
import json
from torch.utils import data
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from config_xlnet_bilstm import Config
import torch.nn.functional as F1
import torch.nn as nn
import time
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from datetime import timedelta
from tensorboardX import SummaryWriter
from torch.functional import F
import numpy as np
from sklearn import metrics
import pickle
config = Config()
device = config.device
print(torch.cuda.is_available())

device_ids=range(torch.cuda.device_count())
import torch.nn
#tokenizer = XLNetTokenizer.from_pretrained('embd/chinese-xlnet-mid')
#xlnet = XLNetModel.from_pretrained('embd/chinese-xlnet-mid').to(device)
# xlnet = XLNetLMHeadModel.from_pretrained('xlnet-base-cased'', mem_len=1024)




def un_fenci(file1,file2):
        f = open(file1,'r',encoding='utf-8')
        fr = open(file2,'w',encoding='utf-8')
        lines1 = f.readlines()
        ll=[]
        for line in lines1:
            line = json.loads(line)
            title = line['HEADLINE']
            text = line['TEXT']
            score = line['score']
            new_sent = []
            for item in text:
                s = ''
                for word in item:
                     s+=word
                new_sent.append(s)
            fr.write(str(title)+'\t'+str(new_sent)+'\t'+str(score)+'\n')
#un_fenci('data/res_data.json','data/un_data.txt')

def get_essay_representation(file,s):
    outputs = []
    input_sents = []
    EDU_breaks = []
    f = open(file,'r',encoding='utf-8')
    lines = f.readlines()
    ll = []
    #if s == 'train':
        #lines = lines[:500]
    #if s == 'dev':
        #lines = lines[:200]
    #if s == 'test':
        #lines = lines[:200]
    #print('len_lines',len(lines))
    for line in lines:
        line = eval(line.strip().split('\t')[1])
        input_sent = line
        input = []
        EDUbreak = []
        lenss = []
        for i in range(len(input_sent)):
            lenss.append(len(tokenizer.tokenize(input_sent[i])))
            input.extend(tokenizer.tokenize(input_sent[i]))
            input.append('SEP')
            EDUbreak.append(len(input)-1)
        EDU_breaks.append(EDUbreak)
        input.append('CLS')
        input_sents.append(input)
    print(len(input_sents))
    for item in input_sents:
        out_isd = tokenizer.convert_tokens_to_ids(item)
        # print(out_isd)
        token_tensor = torch.tensor([out_isd]).to(device)
        sentenc_vector = xlnet(token_tensor)
        outputs.append(sentenc_vector[0].cpu().detach().numpy().tolist())
    print(len(outputs))

    max1 = 0
    for i in range(len(outputs)):
        # print(i)
        if len(outputs[i][0])>max1:
            max1 = len(outputs[i][0])
    # print(max1)

    for i in range(len(outputs)):
        if len(outputs[i][0])<max1:
            for k in range(max1-len(outputs[i][0])):
                outputs[i][0].append([0]*768)
        print('kk',s,len(outputs[i][0]))

    input_embedding = torch.tensor(outputs)
    input_embedding = torch.squeeze(input_embedding)
    print(input_embedding.shape)
    return input_embedding,EDU_breaks

#all_embedding,all_Edu_breaks = get_essay_representation('data/un_data.txt','test')
'''句子平均池化'''
def get_emb(EncoderOutputs,EDU_breaks):
    res = []
    print('1',len(EDU_breaks))
    for i in range(len(EDU_breaks)):
        print('2',EDU_breaks[i])
        lst = []
        if EDU_breaks[i][0]==0:  # j为每个edu跨度
            EDU_breaks[i]= EDU_breaks[i][1:]
        print('2',EDU_breaks[i])
        for j in range(len(EDU_breaks[i])):  # j为每个edu跨度
            lst1 = []
            if j == 0:
                for x in range(EDU_breaks[i][0]):
                    lst1.append(EncoderOutputs[i][x].tolist())
                ten = torch.tensor(lst1)
                teh = ten[-1:].squeeze(0)
                # print('teh', teh)
            else:
                for x in range(EDU_breaks[i][j-1]+1,EDU_breaks[i][j]+1):
                    lst1.append(EncoderOutputs[i][x].tolist())
                ten = torch.tensor(lst1)
                #print('ten',ten[-1:].shape)
                teh = ten[-1:].squeeze(0)
                #print(teh.shape)
            #print('teh', len(teh))
            lst.append(teh.tolist())
        #print('wow')
        #print(len(lst))
        if len(lst)<20:
            lst+=[[0]*768]*(27-len(lst))
        #print(type(lst))
        ten2 = torch.Tensor(lst)
        cur_EncoderOutputs = ten2
        print('ten2',ten2.shape)
        res.append(cur_EncoderOutputs)
    return res
#all_embedding = get_emb(all_embedding,all_Edu_breaks)
#print('all_embedding',len(all_embedding))
d = {'Bad':0,'Medium':1,'Great':2}
class DataGen(data.Dataset):
    def __init__(self,datas,emb,s,scores):
        self.datas = datas
        self.s = s
        self.vob1 = json.load(open('vob_pf.json','r',encoding='utf-8'))
        self.emb = emb
        self.data = []
        self.func = []
        self.scores = scores
        self.sent_func = []
        self.label = []#label
        self.stru_label = []
        self.topic_label = []
        self.logic_label = []
        self.lang_label = []
        self.get_data_label_func()


    def read_file(self,file):
        with open(file,'r',encoding='utf8')as fr:
            lines = fr.readlines()
        return lines
    def get_data_label_func(self):
        #if self.s=='train':
            #lines = lines[:500]
        #if self.s == 'test':
            #lines = lines[:200]
        for i in range(len(self.datas)):
            item = eval(self.datas[i])
            self.data.append(self.emb[i][:20])
            self.label.append(d[self.scores[i]["score"]])
            self.stru_label.append(d[self.scores[i]["stru_score"]])
            self.topic_label.append(d[self.scores[i]["topic_score"]])
            self.logic_label.append(d[self.scores[i]["logic_score"]])
            self.lang_label.append(d[self.scores[i]["lang_score"]])
            word_2_idx_fun = self.trans_word_list_to_func(item['paras'], self.vob1, 20)
            res = []
            for j in range(len(item['paras'])):
                res.append([item['labels'][k] for k in range(len(item['pid'])) if
                            item['pid'][k] == j + 1])
            self.sent_func.append(self.trans_sfunc_to_list(res))
            self.func.append(word_2_idx_fun)
    def trans_word_list_to_func(self,word_list,vob,num):
        """

        :param word_list: ['的'，’的‘]
        :return: [idx1,idx2]
        """
        if len(word_list)<=num:
            word_list+=['<pad>']*(num-len(word_list))
        else:
            word_list=word_list[:num]
        idx_list = []
        for word in word_list:
            idx_list.append(vob[word])
        return idx_list
    def trans_sfunc_to_list(self,func):
        s_func_d = json.load(open('vob_sf.json','r',encoding='utf-8'))
        res = []
        for item in func:
            s_res = []
            for f in item:
                s_res.append(s_func_d[f])
            if len(s_res)<=20:
                s_res+=[0]*(20-len(s_res))
            else:
                s_res=s_res[:20]
            res.append(s_res)
        if len(res)<=20:
            res+=[[0]*20]*(20-len(res))
        else:
            res = res[:20]
        return res
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index],self.label[index],self.func[index],self.sent_func[index],self.stru_label[index],self.topic_label[index],self.logic_label[index],self.lang_label[index]


def collote_fn(batch_seq):
    batch_label = []
    batch_idx = []
    batch_func = []
    batch_sent_func = []
    batch_stru_label = []
    batch_topic_label = []
    batch_logic_label = []
    batch_lang_label = []
    for x in batch_seq:
        batch_idx.append(x[0])
        batch_label.append(x[1])
        batch_func.append(x[2])
        batch_sent_func.append(x[3])
        batch_stru_label.append(x[4])
        batch_topic_label.append(x[5])
        batch_logic_label.append(x[6])
        batch_lang_label.append(x[7])
    batch_idx = torch.Tensor(batch_idx).to(device)  # 改动
    # print('ff',batch_idx)
    batch_label = torch.LongTensor(batch_label).to(device)
    batch_func = torch.LongTensor(batch_func).to(device)
    batch_sent_func = torch.LongTensor(batch_sent_func).to(device)
    batch_stru_label = torch.LongTensor(batch_stru_label).to(device)
    batch_topic_label = torch.LongTensor(batch_topic_label).to(device)
    batch_logic_label = torch.LongTensor(batch_logic_label).to(device)
    batch_lang_label = torch.LongTensor(batch_lang_label).to(device)

    return (batch_idx,batch_label,batch_func,batch_sent_func,batch_stru_label,batch_topic_label,batch_logic_label,batch_lang_label)

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(768*20, config.num_classes)
        self.tanh = torch.nn.Tanh()

    def forward(self,idx):

        #print('idx',idx.shape)
        batch_n, doc_l, emb = idx.size()
        idx = idx.view(batch_n,doc_l*emb)
        #idx = self.dropout(idx)
        out = self.fc(idx)  # 句子最后时刻的 hidden state
        #print(out.shape)
        return out


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, train_iter, test_iter,dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    loss_MAE = torch.nn.L1Loss()
    loss_MSE = torch.nn.MSELoss()
    dev_best_qwk = float('-inf')
    iter = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels,funcs,sent_func,stru_label, topic_label, logic_label, lang_label) in enumerate(train_iter):
            # print('de:',feat1,feat2,feat3,feat4,feat5)
            #print(trains,labels,lang_label)
            outputs = model(trains)
            # print(outputs)
            optimizer.zero_grad()
            loss = F.cross_entropy(outputs,lang_label)
            loss.backward()
            optimizer.step()
            iter+=1
            if iter % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = lang_label.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                train_MSE = metrics.mean_squared_error(true,predic)
                train_MAE = metrics.mean_absolute_error(true, predic)
                train_QWK = metrics.cohen_kappa_score(predic, true, weights='quadratic')
                dev_qwk, dev_loss = evaluate(config, model, dev_iter)
                if dev_qwk >= dev_best_qwk:
                    dev_best_qwk = dev_qwk
                    torch.save(model.state_dict(), config.save_path+'{}.ckpt'.format(iter))
                    test(config, model, test_iter)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train QWK: {2:>6.2%}, Train MAE: {3:>6.2%},Train MSE: {4:>6.2%}, Val Loss: {5:>5.2},  Val QWK: {6:>6.2%},  Time: {7}'
                print(msg.format(iter, loss.item(), train_QWK,train_MAE.item(),train_MSE.item(),  dev_loss, dev_qwk, time_dif))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("qwk/train", train_QWK, total_batch)
                writer.add_scalar("qwk/dev", dev_qwk, total_batch)
                model.train()
    writer.close()
    test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    f = open('result_xlnet/result_lang4.txt', 'a', encoding='utf-8')
    f.write('这是第4折')
    model.eval()
    start_time = time.time()
    test_qwk, test_loss, test_report, test_confusion,test_MAE,test_MSE = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test QWK: {1:>6.2%},Test MAE: {2:>6.2%},Test MSE: {3:>6.2%}'
    print(msg.format(test_loss, test_qwk,test_MAE,test_MSE))
    f.write(msg.format(test_loss, test_qwk,test_MAE,test_MSE))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    f.write(test_report)
    print("Confusion Matrix...")
    print(test_confusion)
    f.write(str(test_confusion)+'\n')
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels,funcs,sent_func,stru_label, topic_label, logic_label, lang_label in data_iter:
            # print('dff:',feat1,feat2,feat3,feat4,feat5)

            outputs = model(texts)
            loss = F.cross_entropy(outputs, lang_label)
            loss_total += loss
            labels = (lang_label).data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)

    qwk = metrics.cohen_kappa_score(predict_all, labels_all, weights='quadratic')
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        test_MSE = metrics.mean_squared_error(labels_all, predict_all)
        test_MAE = metrics.mean_absolute_error(labels_all, predict_all)
        return qwk, loss_total / len(data_iter), report, confusion,test_MAE.item(),test_MSE.item()
    return qwk, loss_total / len(data_iter)


config = Config()
def getRandomIndex(n, x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(n, size=x, replace=False)
    return index

f = open('data/cv_folds.txt', 'r', encoding='utf-8')
import numpy as np
lines = f.readlines()
i = 0
f = open('data/al_embedding.txt','r',encoding='utf-8')
all_embedding = f.read()
all_embedding = eval(all_embedding)
model = Model(config)
model = model.to(config.device)
line = lines[4]
line = line.strip().split('\t')
line = [int(item) for item in line]
all = list(range(1220))
test_data_index = line
train_data_index = [item for item in all if item not in test_data_index]
dev_data_index = getRandomIndex(train_data_index, 98)
dev_data_index = list(dev_data_index)
train_data_index = [item for item in train_data_index if item not in dev_data_index]#20 *20
f = open('data/all_data.json', 'r', encoding='utf-8')
lines = f.readlines()
train_data = [lines[i] for i in train_data_index]
all_emb = np.array(all_embedding)
train_emb = all_emb[train_data_index]
test_data = [lines[i] for i in test_data_index]
test_emb = all_emb[test_data_index]
dev_data = [lines[i] for i in dev_data_index]
dev_emb = all_emb[dev_data_index]
f = open('data/score_ult.json', 'r', encoding='utf-8')
lines = f.readlines()
train_score = [json.loads(lines[i]) for i in train_data_index]
test_score = [json.loads(lines[i]) for i in test_data_index]
dev_score = [json.loads(lines[i]) for i in dev_data_index]
train_data = DataGen(train_data,train_emb, 'vob.json',train_score)
test_data = DataGen(test_data,test_emb, 'vob.json',test_score)
dev_data = DataGen(dev_data,dev_emb, 'vob.json',dev_score)
train_sampler = RandomSampler(train_data)
test_sampler = SequentialSampler(test_data)
dev_sampler = SequentialSampler(dev_data)
train_data_loader = DataLoader(train_data, batch_size=config.batch_size, sampler=train_sampler,
                               collate_fn=collote_fn, drop_last=False)
test_data_loader = DataLoader(test_data, batch_size=config.batch_size, sampler=test_sampler, collate_fn=collote_fn,
                              drop_last=False)
dev_data_loader = DataLoader(dev_data, batch_size=config.batch_size, sampler=dev_sampler, collate_fn=collote_fn,
                             drop_last=False)
print('这是第4折')

train(config, model, train_data_loader, test_data_loader, dev_data_loader)
i += 1

