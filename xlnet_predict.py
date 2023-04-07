from pytorch_transformers import XLNetTokenizer,XLNetModel,XLNetForSequenceClassification
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
# from data_process import get_embedding
from multi_head import MultiHeadAttention
import pickle
from tqdm import tqdm
config = Config()
device = config.device
print(torch.cuda.is_available())

device_ids=range(torch.cuda.device_count())
import torch.nn
# tokenizer = XLNetTokenizer.from_pretrained('embd/chinese-xlnet-mid')
# xlnet = XLNetModel.from_pretrained('embd/chinese-xlnet-mid').to(device)
# xlnet = XLNetLMHeadModel.from_pretrained('xlnet-base-cased'', mem_len=102
import math
def calcMean(x,y):
    sum_x = sum(x)
    sum_y = sum(y)
    n = len(x)
    x_mean = float(sum_x+0.0)/n
    y_mean = float(sum_y+0.0)/n
    return x_mean,y_mean

#计算Pearson系数
def calcPearson(x,y):
    x_mean,y_mean = calcMean(x,y)	#计算x,y向量平均值
    n = len(x)
    sumTop = 0.0
    sumBottom = 0.0
    x_pow = 0.0
    y_pow = 0.0
    for i in range(n):
        sumTop += (x[i]-x_mean)*(y[i]-y_mean)
    for i in range(n):
        x_pow += math.pow(x[i]-x_mean,2)
    for i in range(n):
        y_pow += math.pow(y[i]-y_mean,2)
    sumBottom = math.sqrt(x_pow*y_pow)
    p = sumTop/sumBottom
    return p
d = {'Bad':0,'Medium':1,'Great':2}
d_func = {"OtherPara":1, "SupportPara":2, "IdeaPara":3,"ConclusionPara": 4,"IntroductionPara": 5,"ThesisPara": 6}
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

class Model_sent(nn.Module):
    def __init__(self):
        super(Model_sent, self).__init__()

        self.fc = nn.Linear(768, 2)  ##全连接层
        self.tanh = nn.Tanh()
    def forward(self, x):
        out = self.fc(x)
        # out  = self.tanh(out)

        return out
config = Config()
device = config.device
# model_sent = Model_sent()
# model_sent.load_state_dict(torch.load('saved_dict/predict_sent970.ckpt'))
# model_sent = model_sent.to(config.device)
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
    dev_best_loss = float('inf')
    dev_best_QWK = float('-inf')
    dev_best_QWK1= float('-inf')
    dev_best_QWK2 = float('-inf')
    dev_best_QWK3 = float('-inf')
    dev_best_QWK4 = float('-inf')

    iter = 0
    writer = SummaryWriter(log_dir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels,funcs,sent_func,stru_label, topic_label, logic_label, lang_label) in enumerate(train_iter):
            # print('de:',feat1,feat2,feat3,feat4,feat5)
            #print(trains,labels)
            outputs,outputs1,outputs2,outputs3,outputs4 = model(trains)
            optimizer.zero_grad()
            loss0 = F.cross_entropy(outputs, labels)
            loss1 = F.cross_entropy(outputs1, stru_label)
            loss2 = F.cross_entropy(outputs2, topic_label)
            loss3 = F.cross_entropy(outputs3, logic_label)
            loss4 = F.cross_entropy(outputs4, lang_label)
            loss = (loss0 + loss2 + loss1 + loss3 + loss4)/4
            loss.backward()
            optimizer.step()
            iter+=1
            if iter % 10 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                predic = torch.max(outputs.data, 1)[1].cpu()
                true1 = stru_label.data.cpu()
                predic1 = torch.max(outputs1.data, 1)[1].cpu()
                true2 = topic_label.data.cpu()
                predic2 = torch.max(outputs2.data, 1)[1].cpu()
                true3 = logic_label.data.cpu()
                predic3 = torch.max(outputs3.data, 1)[1].cpu()
                true4 = lang_label.data.cpu()
                predic4 = torch.max(outputs4.data, 1)[1].cpu()
                train_QWK = metrics.cohen_kappa_score(predic, true, weights='quadratic')
                train_QWK1 = metrics.cohen_kappa_score(predic1, true1, weights='quadratic')
                train_QWK2 = metrics.cohen_kappa_score(predic2, true2, weights='quadratic')
                train_QWK3 = metrics.cohen_kappa_score(predic3, true3, weights='quadratic')
                train_QWK4 = metrics.cohen_kappa_score(predic4, true4, weights='quadratic')

                print(train_QWK)
                # train_MAE = metrics.mean_absolute_error(true, predic)
                # train_acc = metrics.accuracy_score(true, predic)
                dev_QWK, dev_QWK_stru,dev_QWK_topic,dev_QWK_logic, dev_QWK_lang,dev_loss = evaluate(config, model, dev_iter)
                if dev_QWK >= dev_best_QWK or dev_QWK_stru>= dev_best_QWK1 or dev_QWK_logic>= dev_best_QWK3 or dev_QWK_topic>= dev_best_QWK2 or dev_QWK_lang >= dev_best_QWK4 or dev_loss<=dev_best_loss:
                    if dev_QWK >= dev_best_QWK:
                        dev_best_QWK = dev_QWK
                    if dev_QWK_stru>= dev_best_QWK1:
                        dev_best_QWK1 = dev_QWK_stru
                    if dev_QWK_topic>= dev_best_QWK2:
                        dev_best_QWK2 = dev_QWK_topic
                    if dev_QWK_logic >= dev_best_QWK3:
                        dev_best_QWK3 = dev_QWK_logic
                    if dev_QWK_lang>= dev_best_QWK4:
                        dev_best_QWK4 = dev_QWK_lang
                    if dev_loss <= dev_best_loss:
                        dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path+'{}.ckpt'.format(iter))
                    test(config, model, test_iter)
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train QWK: {2:>6.2%},Train QWK1: {3:>6.2%},Train QWK2: {4:>6.2%},Train QWK3: {5:>6.2%},Train QWK4: {6:>6.2%},,Val Loss: {7:>5.2},  Val QWK: {8:>6.2%},Val QWK_stru: {9:>6.2%},Val QWK_topic: {10:>6.2%},Val QWK_logic: {10:>6.2%},Val QWK_lang: {11:>6.2%},Time: {12}'
                print(msg.format(iter, loss.item(), train_QWK,train_QWK1,train_QWK2,train_QWK3,train_QWK4, dev_loss, dev_QWK,dev_QWK_stru,dev_QWK_topic,dev_QWK_logic,dev_QWK_lang, time_dif))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("QWK/train", train_QWK, total_batch)
                writer.add_scalar("QWK_stru/train", train_QWK1, total_batch)
                writer.add_scalar("QWK_topic/train", train_QWK2, total_batch)
                writer.add_scalar("QWK_logic/train", train_QWK3, total_batch)
                writer.add_scalar("QWK_lang/train", train_QWK4, total_batch)
                writer.add_scalar("QWK/dev", dev_QWK, total_batch)
                writer.add_scalar("QWK_stru/dev", dev_QWK_stru, total_batch)
                writer.add_scalar("QWK_topic/dev", dev_QWK_topic, total_batch)
                writer.add_scalar("QWK_logic/dev", dev_QWK_logic, total_batch)
                writer.add_scalar("QWK_lang/dev", dev_QWK_lang, total_batch)
                model.train()
    writer.close()
    test(config, model, test_iter)

def test(config, model, test_iter):
    # test
    f = open('result/new_test/result_xlnet_pre.txt', 'a', encoding='utf-8')
    f.write('这是第4折')
    model.eval()
    start_time = time.time()
    test_qwk,test_qwk_stru,test_qwk_topic,test_qwk_logic,test_qwk_lang, test_loss, test_report, test_confusion, test_report_stru, test_confusion_stru, test_report_topic, test_confusion_topic, test_report_logic, test_confusion_logic, test_report_lang, test_confusion_lang= evaluate(config, model, test_iter, test=True)
    # print('test',test_qwk)
    msg = 'Test Loss: {0:>5.2},  Test QWK: {1:>6.2%},Test QWK_stru: {2:>6.2%},Test QWK_topic: {3:>6.2%},Test QWK_Logic: {3:>6.2%},Test QWK_Lang: {4:>6.2%}'
    print(msg.format(test_loss, test_qwk,test_qwk_stru,test_qwk_topic,test_qwk_logic,test_qwk_lang))
    f.write(msg.format(test_loss, test_qwk,test_qwk_stru,test_qwk_topic,test_qwk_logic,test_qwk_lang))
    print("Precision, Recall and F1-Score...")
    print(test_report)
    f.write(test_report)
    print(test_report_stru)
    f.write(test_report_stru)
    print(test_report_topic)
    f.write(test_report_topic)
    print(test_report_logic)
    f.write(test_report_logic)
    print(test_report_lang)
    f.write(test_report_lang)
    print("Confusion Matrix...")
    print(test_confusion)
    f.write(str(test_confusion)+'\n')
    print(test_confusion_stru)
    f.write(str(test_confusion_stru)+'\n')
    print(test_confusion_topic)
    f.write(str(test_confusion_topic)+'\n')
    print(test_confusion_logic)
    f.write(str(test_confusion_logic)+'\n')
    print(test_confusion_lang)
    f.write(str(test_confusion_lang)+'\n')
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    predict_all1 = np.array([], dtype=int)
    labels_all1 = np.array([], dtype=int)
    predict_all2 = np.array([], dtype=int)
    labels_all2 = np.array([], dtype=int)
    predict_all3 = np.array([], dtype=int)
    labels_all3 = np.array([], dtype=int)
    predict_all4 = np.array([], dtype=int)
    labels_all4 = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels,funcs,sent_func,stru_label, topic_label, logic_label, lang_label in data_iter:
            outputs,outputs1,outputs2,outputs3,outputs4 = model(texts)
            loss0 = F.cross_entropy(outputs, labels)
            loss1 = F.cross_entropy(outputs1, stru_label)
            loss2 = F.cross_entropy(outputs2, topic_label)
            loss3 = F.cross_entropy(outputs3, logic_label)
            loss4 = F.cross_entropy(outputs4, lang_label)
            loss = (loss0 + loss2 + loss1 + loss3 +loss4 )/4
            loss_total += loss
            labels = (labels).data.cpu().numpy()
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels1 = (stru_label).data.cpu().numpy()
            predic1 = torch.max(outputs1.data, 1)[1].cpu().numpy()
            labels2 = (topic_label).data.cpu().numpy()
            predic2 = torch.max(outputs2.data, 1)[1].cpu().numpy()
            labels3 = (logic_label).data.cpu().numpy()
            predic3 = torch.max(outputs3.data, 1)[1].cpu().numpy()
            labels4 = (lang_label).data.cpu().numpy()
            predic4 = torch.max(outputs4.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
            labels_all1 = np.append(labels_all1, labels1)
            predict_all1 = np.append(predict_all1, predic1)
            labels_all2 = np.append(labels_all2, labels2)
            predict_all2 = np.append(predict_all2, predic2)
            labels_all3 = np.append(labels_all3, labels3)
            predict_all3 = np.append(predict_all3, predic3)
            labels_all4 = np.append(labels_all4, labels4)
            predict_all4 = np.append(predict_all4, predic4)

    # acc = metrics.accuracy_score(labels_all, predict_all)
    qwk = metrics.cohen_kappa_score(predict_all, labels_all, weights='quadratic')
    qwk1 = metrics.cohen_kappa_score(predict_all1, labels_all1, weights='quadratic')
    qwk2 = metrics.cohen_kappa_score(predict_all2, labels_all2, weights='quadratic')
    qwk3 = metrics.cohen_kappa_score(predict_all3, labels_all3, weights='quadratic')
    qwk4 = metrics.cohen_kappa_score(predict_all4, labels_all4, weights='quadratic')
    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.class_list, digits=4)
        report_stru = metrics.classification_report(labels_all1, predict_all1, target_names=config.class_list, digits=4)
        report_topic = metrics.classification_report(labels_all2, predict_all2, target_names=config.class_list, digits=4)
        report_logic = metrics.classification_report(labels_all3, predict_all3, target_names=config.class_list, digits=4)
        report_lang = metrics.classification_report(labels_all4, predict_all4, target_names=config.class_list, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        confusion_stru = metrics.confusion_matrix(labels_all1, predict_all1)
        confusion_topic = metrics.confusion_matrix(labels_all2, predict_all2)
        confusion_logic = metrics.confusion_matrix(labels_all3, predict_all3)
        confusion_lang = metrics.confusion_matrix(labels_all4, predict_all4)

        # test_MSE = metrics.mean_squared_error(labels_all, predict_all)
        # test_MAE = metrics.mean_absolute_error(labels_all, predict_all)
        return qwk,qwk1,qwk2,qwk3,qwk4, loss_total / len(data_iter), report, confusion,report_stru,confusion_stru,report_topic,confusion_topic,report_logic,confusion_logic,report_lang,confusion_lang
    return qwk,qwk1,qwk2,qwk3,qwk4,loss_total / len(data_iter)


config = Config()
def getRandomIndex(n, x):
    # 索引范围为[0, n)，随机选x个不重复，注意replace=False才是不重复，replace=True则有可能重复
    index = np.random.choice(n, size=x, replace=False)
    return index
def get_emb(file_path):
    f = open(file_path,'r',encoding='utf-8') #'data/al_sent_emb.txt'
    all_embedding = f.read()
    all_embedding = eval(all_embedding)
    return all_embedding
def process(emb):
    lines = open('data/all_data.json','r',encoding='utf-8').readlines()
    embs = []
    for i in range(len(lines)):
        pid = json.loads(lines[i])['pid']
        k = 1
        ress = []
        res = []
        for j in range(len(emb[i])):
            if pid[j]==k:
                res.append(emb[i][j])
            else:
                k+=1
                ress.append(res)
                res = []
        for i in range(len(ress)):
            if len(ress[i])<20:
                ress[i]+=[[0]*768]*(20-len(ress[i]))
            else:
                ress[i] = ress[i][:20]
        if len(ress)<20:
            ress+=[20*[[0]*768]]*(20-len(ress))
        else:
            ress = ress[:20]
        embs.append(ress)
    return embs

def get_para_emb(file_path):
    f = open(file_path,'r',encoding='utf-8') #'data/al_sent_emb.txt'
    all_embedding = f.read()
    all_embedding = eval(all_embedding)
    for i in range(len(all_embedding)):
        if len(all_embedding[i]) <21:
            all_embedding[i] = [[0]*768]*(21-len(all_embedding[i]))
        else:
            all_embedding[i] = all_embedding[i][:21]
    return all_embedding
def get_embedding():
    f = open('data/al_embedding.txt','r',encoding='utf-8')
    # f.write(str(all_embedding))
    f1 = open('data/para_sent.txt','r',encoding='utf-8')
    # f1.write(str(para_len))
    para_sent = eval(f1.read())
    emb = eval(f.read())
    p = []
    for i in range(len(para_sent)):
        para_i = []
        para_i.append([emb[i][0]])
        for j in range(len(para_sent[i])):
            e = emb[i][sum(para_sent[i][:j])+1:sum(para_sent[i][:j+1])+1]
            if len(e)!=768:
                para_i.append(e)
            else:
                para_i.append([e])
        p.append(para_i)
    for i in range(len(p)):
        for j in range(1,len(p[i])):
            if len(p[i][j])<20:
                p[i][j]+=[[0]*768]*(20-len(p[i][j]))
        if len(p[i])<21:
            p[i] +=[20*[[0]*768]]*(21-len(p[i]))
        else:
            p[i] = p[i][:21]
    title = []
    para = []
    for item in p:
        print(len(item))
        title.append(item[0])
        para.append(item[1:])
        print(len(title),len(para))
    return torch.Tensor(title).squeeze(dim=1), torch.Tensor(para)

def main():
    f = open('data/cv_folds.txt', 'r', encoding='utf-8')
    import numpy as np
    lines = f.readlines()
    i = 0
    f = open('data/al_embedding.txt', 'r', encoding='utf-8')
    all_embedding = f.read()
    all_embedding = eval(all_embedding)
    model = Model(config)
    model = model.to(config.device)
    line = lines[4]
    line = line.strip().split('\t')[0]
    print(line)
    line = [int(float(item)) for item in eval(line)]
    all = list(range(1220))
    test_data_index = line
    train_data_index = [item for item in all if item not in test_data_index]
    dev_data_index = getRandomIndex(train_data_index, 98)
    dev_data_index = list(dev_data_index)
    train_data_index = [item for item in train_data_index if item not in dev_data_index]  # 20 *20
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
    train_data = DataGen(train_data, train_emb, 'vob.json', train_score)
    test_data = DataGen(test_data, test_emb, 'vob.json', test_score)
    dev_data = DataGen(dev_data, dev_emb, 'vob.json', dev_score)
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

    # train(config, model, train_data_loader, test_data_loader, dev_data_loader)
    # model_predict = Model(config)
    # model_predict.load_state_dict(torch.load('saved_dict/rebuttle470.ckpt'))
    # model_predict = model_predict.to(config.device)
    # print(test_emb[0])
    # f = open('res.txt','w',encoding='utf-8')
    # for i in range(len(test_emb)):
    #     data = test_emb[i].to(config.device)
    #     # t = title[i].to(config.device)
    #     output,output1,output2,output3,output4 = model_predict(data)
    #     predic = torch.max(output.data, 1)[1].cpu().numpy()
    #     predic1 = torch.max(output1.data, 1)[1].cpu().numpy()
    #     predic2 = torch.max(output2.data, 1)[1].cpu().numpy()
    #     predic3 = torch.max(output3.data, 1)[1].cpu().numpy()
    #     predic4 = torch.max(output4.data, 1)[1].cpu().numpy()
    #     f.write(test_score[i]+'\t'+predic+'\t'+predic1+'\t'+predic2+'\t'+predic3+'\t'+predic4+'\n')
    i += 1
main()
def tt():
    model = Model(config)
    model = model.to(config.device)
    model.load_state_dict(torch.load('saved_dict/new_test710.ckpt'))#xlnet_bilstm_80_0.02790
    f_test= open('data/emb.txt','r',encoding='utf-8').readlines()
    print(isinstance(f_test[0],str))
    f_test_data = eval(f_test[0])
    f_test_data = torch.Tensor(f_test_data).to(device)
    print(f_test_data.shape)
    f_res = open('data/result.txt', 'a', encoding='utf-8')
    f_val = open('data/test.json', 'r', encoding='utf-8')
    lines = f_val.readlines()
    for i in range(len(lines)):
        outputs = model(f_test_data[i].unsqueeze(dim=0))
        predic = torch.max(outputs.data, 1)[1].cpu()
        print(predic)
        d = {}
        t = json.loads(lines[i])
        d["url"] = t["url"]
        d["label"] = predic
        f_res.write(str(d)+'\n')

# tt()