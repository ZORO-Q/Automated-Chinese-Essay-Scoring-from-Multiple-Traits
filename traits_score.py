#Essays traits
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
d = {'Bad':0,'Medium':1,'Great':2}
d_func = {"OtherPara":1, "SupportPara":2, "IdeaPara":3,"ConclusionPara": 4,"IntroductionPara": 5,"ThesisPara": 6}
class DataGen(data.Dataset):
    def __init__(self,scores,emb,title):
        self.scores = scores
        self.emb = emb
        self.title = title
        self.titles = []
        self.data = []
        self.datasen = []
        self.stru_label = []
        self.topic_label = []
        self.logic_label = []
        self.lang_label = []
        self.label = []
        self.get_data_label_func()

    def get_data_label_func(self):
        for i in range(len(self.scores)):
            self.data.append(self.emb[i].numpy())
            self.titles.append(self.title[i].numpy())
            self.stru_label.append(d[self.scores[i]["stru_score"]])
            self.topic_label.append(d[self.scores[i]["topic_score"]])
            self.logic_label.append(d[self.scores[i]["logic_score"]])
            self.lang_label.append(d[self.scores[i]["lang_score"]])
            self.label.append(d[self.scores[i]["score"]])


    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.data[index],self.label[index],self.stru_label[index],self.topic_label[index],self.logic_label[index],self.lang_label[index],self.titles[index]

def collote_fn(batch_seq):
    batch_idx = []
    batch_label = []
    batch_stru_label = []
    batch_topic_label = []
    batch_logic_label = []
    batch_lang_label = []
    batch_title = []

    for x in batch_seq:
        batch_idx.append(x[0])
        batch_label.append(x[1])
        batch_stru_label.append(x[2])
        batch_topic_label.append(x[3])
        batch_logic_label.append(x[4])
        batch_lang_label.append(x[5])
        batch_title.append(x[6])
    batch_idx = torch.Tensor(batch_idx).to(device)  # 改动
    batch_label = torch.LongTensor(batch_label).to(device)
    batch_stru_label = torch.LongTensor(batch_stru_label).to(device)
    batch_topic_label = torch.LongTensor(batch_topic_label).to(device)
    batch_logic_label = torch.LongTensor(batch_logic_label).to(device)
    batch_lang_label = torch.LongTensor(batch_lang_label).to(device)
    batch_title = torch.Tensor(batch_title).to(device)

    return (batch_idx,batch_label,batch_stru_label,batch_topic_label,batch_logic_label,batch_lang_label,batch_title)

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
class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.p_emb = nn.Embedding(7,16,padding_idx=0)
        self.emo_emb = nn.Embedding(3,16,padding_idx=0)
        self.le_emb = nn.Embedding(28,16)
        self.posit_emb = nn.Embedding(11,16)
        self.neg_emb = nn.Embedding(11,16)
        self.w_omega = nn.Parameter(torch.Tensor(
            config.hidden_size*2 , config.hidden_size*2))
        self.u_omega = nn.Parameter(torch.Tensor(config.hidden_size*2, 1))
        self.w_omega_t = nn.Parameter(torch.Tensor(
            config.hidden_size * 2, config.hidden_size * 2))
        self.u_omega_t = nn.Parameter(torch.Tensor(config.hidden_size * 2, 1))
        nn.init.uniform_(self.w_omega, -0.1, 0.1)
        nn.init.uniform_(self.u_omega, -0.1, 0.1)
        nn.init.uniform_(self.w_omega_t, -0.1, 0.1)
        nn.init.uniform_(self.u_omega_t, -0.1, 0.1)
        self.lstm = nn.LSTM(config.hidden_size*2, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm_stru = nn.LSTM(config.hidden_size * 2, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm_top = nn.LSTM(config.hidden_size * 2, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm_log = nn.LSTM(config.hidden_size * 2, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm_lang = nn.LSTM(config.hidden_size * 2, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.lstm1 = nn.LSTM(config.embd, config.hidden_size, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self.p_func_lstm = nn.LSTM(32,32, config.num_layers,
                             bidirectional=True, batch_first=True, dropout=config.dropout)
        self.Relation = nn.Linear(5,5)
        self.attention = MultiHeadAttention(config.hidden_size*2*5,8,0.2)
        self.f_fea = nn.Linear(368, config.hidden_size //2)
        self.fcs_0 = nn.Linear(config.hidden_size * 10, config.hidden_size * 2)
        self.fcs_1 = nn.Linear(config.hidden_size * 10, config.hidden_size * 2)
        self.fcs_2 = nn.Linear(config.hidden_size * 10, config.hidden_size * 2)
        self.fcs_3 = nn.Linear(config.hidden_size * 10, config.hidden_size * 2)
        self.fcs_4 = nn.Linear(config.hidden_size * 10, config.hidden_size * 2)
        self.fcs0 = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.fcs1 = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.fcs2 = nn.Linear(config.hidden_size * 2+64, config.num_classes)
        self.fcs3 = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.fcs4 = nn.Linear(config.hidden_size * 2, config.num_classes)
        self.fc_res = nn.Linear(100, 100)
        self.fc_t = nn.Linear(config.embd,64)
        self.fc_t1 = nn.Linear(config.hidden_size*4,config.hidden_size*2)
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()
        self.v = torch.rand((config.batch_size,512, 1), requires_grad=True).to((device))
        self.vv = torch.rand((20, 512, 1), requires_grad=True).to(config.device)
        self.vvv = torch.rand((2, 512, 1), requires_grad=True).to(config.device)
        self.vvvv = torch.rand((14, 512, 1), requires_grad=True).to(config.device)


    def forward(self,idx,idx1):#batch*max_para*max_sent*emb_dim idx1:batch*20

        batch,max_para,max_sent,emb_dim = idx.size()
        idx = idx.view(batch*max_para,max_sent,emb_dim)
        idx,_ = self.lstm1(idx)
        u = torch.tanh(torch.matmul(idx, self.w_omega))
        att = torch.matmul(u, self.u_omega)
        att_score = F.softmax(att, dim=1)
        scored_x = idx * att_score
        scored_x = torch.sum(scored_x,dim = 1)
        scored_x = scored_x.view(batch,max_para,config.hidden_size*2)#batch*para*hidden*2
        idx1 = self.fc_t(idx1)
        idx1 = idx1.repeat(1,max_para,1)

        out,_ = self.lstm(scored_x)
        out_stru,_ = self.lstm_stru(scored_x)
        out_topic,_ = self.lstm_top(scored_x)
        out_logic,_ = self.lstm_log(scored_x)
        out_lang,_ = self.lstm_lang(scored_x)
        out = torch.mean(out,dim = 1).unsqueeze(dim=1)
        out_stru = torch.mean(out_stru,dim = 1).unsqueeze(dim=1)
        out_topic = torch.mean(out_topic,dim = 1).unsqueeze(dim=1)
        out_logic = torch.mean(out_logic,dim = 1).unsqueeze(dim=1)
        out_lang = torch.mean(out_lang,dim = 1).unsqueeze(dim=1)
        ou = torch.cat((out,out_stru,out_topic,out_lang),dim = 1)
        ou = torch.transpose(ou,1,2)
        v = torch.transpose(self.v,1,2)
        vv = torch.transpose(self.vv,1,2)
        vvv = torch.transpose(self.vvv,1,2)
        vvvv = torch.transpose(self.vvvv,1,2)
        if ou.size()[0] == config.batch_size :
            alpha = F.softmax(torch.matmul(v, ou), dim=2)  # 1*512*b2 ,b2*512*20 = 1*20 batch*maxpara,1,4
        elif ou.size()[0] == 20 :
            alpha = F.softmax(torch.matmul(vv, ou), dim=2)  # 1*512*b2 ,b2*512*20 = 1*20
        elif ou.size()[0] == 2 :
            alpha = F.softmax(torch.matmul(vvv, ou), dim=2)  # 1*512*b2 ,b2*512*20 = 1*20
        else:
            alpha = F.softmax(torch.matmul(vvvv, ou), dim=2)  # 1*512*b2 ,b2*512*20 = 1*20
        alpha = torch.transpose(alpha,1,2)
        output = torch.tanh(torch.matmul(ou,alpha))#(768*3)*(3*1)=(768*1)
        output = output.squeeze(dim = 2)

        out_topic=torch.cat((output,idx1),dim = 2)
        out0 = self.fcs0(output)
        out1 = self.fcs1(output)
        out2 = self.fcs2(out_topic)
        out3 = self.fcs3(output)
        out4 = self.fcs4(output)

        out0 = self.tanh(out0)
        out1 = self.tanh(out1)
        out2 = self.tanh(out2)
        out3 = self.tanh(out3)
        out4 = self.tanh(out4)
        return out0,out1,out2,out3,out4



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def train(config, model, train_iter, test_iter,dev_iter):
    start_time = time.time()
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    total_batch = 0  # 记录进行到多少batch
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
        for i, (trains, labels,stru_label,topic_label,logic_label,lang_label,title) in enumerate(train_iter):
            # print('de:',feat1,feat2,feat3,feat4,feat5)
            #print(trains,labels)
            outputs,outputs1,outputs2,outputs3,outputs4 = model(trains,title)
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
    f = open('result/test4.txt', 'a', encoding='utf-8')
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
        for texts, labels,stru_label,topic_label,logic_label,lang_label,title in data_iter:
            outputs,outputs1,outputs2,outputs3,outputs4 = model(texts,title)
            loss0 = F.cross_entropy(outputs, labels)
            loss1 = F.cross_entropy(outputs1, stru_label)
            loss2 = F.cross_entropy(outputs2, topic_label)
            loss3 = F.cross_entropy(outputs3, logic_label)
            loss4 = F.cross_entropy(outputs4, lang_label)
            loss = (loss0 + loss1 + loss2 + loss3 +loss4 )/5
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
    i = 1
    model = Model(config)
    model = model.to(config.device)
    line = lines[4]
    line = eval(line.strip())
    all = list(range(1220))
    test_data_index = [i-1 for i in line]
    train_data_index = [item for item in all if item not in test_data_index]
    dev_data_index = getRandomIndex(train_data_index, 98)
    dev_data_index = list(dev_data_index)
    train_data_index = [item for item in train_data_index if item not in dev_data_index]#20 *20
    f = open('data/score_ult.json', 'r', encoding='utf-8')
    lines = f.readlines()
    title,emb = get_embedding()
    train_score = [json.loads(lines[i]) for i in train_data_index]
    train_emb = emb[train_data_index]
    train_title = title[train_data_index]
    test_score = [json.loads(lines[i]) for i in test_data_index]
    test_emb = emb[test_data_index]
    test_title = title[test_data_index]
    dev_score = [json.loads(lines[i]) for i in dev_data_index]
    dev_emb = emb[dev_data_index]
    dev_title = title[dev_data_index]

    train_data = DataGen(train_score,train_emb,train_title)
    test_data = DataGen(test_score,test_emb,test_title)
    dev_data = DataGen(dev_score,dev_emb,dev_title)
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
main()
